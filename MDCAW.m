function [P,Y_p,Y,mu,R,S,obj] = MDCAW(X,alpha,beta,gamma,sigma,clu_num,maxIter)

opts = optimset('Display','off');

view_num = max(size(X));
dims = zeros(1,view_num);
for v = 1:view_num
	[dims(v),n] = size(X{v});
end



P = cell(1,view_num);
Y_p = cell(1,view_num);
R = cell(1,view_num);
for v = 1:view_num

	R{v} = orth(rand(clu_num,clu_num));
end
clear Temp;

Y = zeros(n,clu_num);

mu = ones(view_num,1)./view_num;
S = zeros(n,n);
obj = zeros(maxIter, 1);

S_v = cell(1,view_num);

for v = 1:view_num
	data_v = X{v}';
	Dis = squareform(pdist(data_v,'squaredeuclidean'));
	Dis = -1.*Dis./sigma;
	S_temp = exp(Dis);
	rowSums = sum(S_temp,2);
	rowSum_v = repmat(rowSums,1,size(S_temp,2));
	S_temp = S_temp./rowSum_v;
	S_v{v} = (S_temp+S_temp')/2;
	
	L = SpectralClustering(S_v{v},clu_num);
    Temp = zeros(n,clu_num);
    for i = 1:n
        Temp(i,L(i)) = 1;
    end
	for j = 1:clu_num
		Temp(:,j) = Temp(:,j)./sqrt(sum(Temp(:,j)));
	end
	Y_p{v} = Temp;
end

clear Dis S_temp rowSum_v rowSums Temp L;

XX = X{1};
for v = 2:view_num
    XX = [XX ;X{v}];

end
data_v = XX';
Dis = squareform(pdist(data_v,'squaredeuclidean'));
Dis = -1.*Dis./sigma;
S_temp = exp(Dis);
rowSums = sum(S_temp,2);
rowSum_v = repmat(rowSums,1,size(S_temp,2));
S_temp = S_temp./rowSum_v;
S = (S_temp+S_temp')/2;
Y_pre = SpectralClustering(S,clu_num);
Temp = zeros(n,clu_num);
for i = 1:n
	Temp(i,Y_pre(i)) = 1;
end
Y = Temp;
clear XX data_v Dis S_temp rowSum_v rowSums S_temp Y_pre Temp;

	



for iter = 1:maxIter
	%%update P
	% disp('update P...');
	for v = 1:view_num
		part_a = X{v}*X{v}'+eye(dims(v)).*alpha;
		P{v} = part_a \ (X{v}*Y_p{v});
	end
	clear part_a;
	
	%%update Y_p
	% disp('update Y_p...');
	for v = 1:view_num
		T = X{v}'*P{v};
		Z = Y;
		for m = 1:view_num
			if m~=v
				Z = Z-Y_p{m}*R{m}.*mu(m);
			end
		end
		B = T + beta.*Z*R{v}';
		LapMatrix = diag(sum(S, 1)) - S;

		Y_p{v} = gpi(LapMatrix.*gamma,B);
	end
	clear T Z LapMatrix;
	
	%%update Y
	% disp('update Y...');
	Y = zeros(n,clu_num);
	A = zeros(n,clu_num);
	for v = 1:view_num
		A = A + Y_p{v}*R{v}.*mu(v);
	end
	for i = 1:n
		[~,k] = max(A(i,:));
		Y(i,k) = 1;
	end
	clear A B;
	

	%%update mu
	% disp('update mu...');
	M = cell(1,view_num);
	Part_p = cell(1,view_num);
	Part_p_new = cell(1,view_num);
	Q = zeros(view_num,view_num);
	Q_new = zeros(view_num,view_num);
	p_temp = zeros(view_num,1);
	p_temp_new = zeros(view_num,1);
	for v = 1:view_num
		M{v} = Y_p{v}*R{v};
	end
	for v = 1:view_num
		Part_p{v} = M{v}*Y';
		Part_p_new{v} = S_v{v}*S';
	end
	p_ = zeros(view_num,1);
	LapMatrix = diag(sum(S, 1)) - S;
	for i = 1:view_num
		for j = 1:view_num
			Temp = M{i}*M{j}';
			Temp2 = S_v{i}*S_v{j}';
			Q(i,j) = trace(Temp)*beta;
			Q_new(i,j) = trace(Temp2);
		end
		p_temp(i) = -trace(Part_p{i})*beta;
		p_temp_new(i) = -trace(Part_p_new{i});
		front = norm(X{i}'*P{i}-Y_p{i},'fro')^2 + alpha.*norm(P{i},'fro')^2 + gamma.*trace(Y_p{i}'*LapMatrix*Y_p{i});
		p_(i) = 0.5*front + p_temp(i) + p_temp_new(i);
	end
	Q = Q + Q_new;
	A = -eye(view_num);
	b = zeros(view_num,1);
	Aeq = ones(1,view_num);
	beq = 1;
	
mu = quadprog(Q,p_,A,b,Aeq,beq,[],[],[],opts);
	clear M Part_p Part_p_new Q Q_new p_temp p_temp_new p_ Temp Temp2 A b Aeq beq;
	
	%%update R
    for v = 1:view_num
		Z = Y;
		for m = 1:view_num
			if m~=v
				Z = Z-Y_p{m}*R{m}.*mu(m);
			end
		end
		Temp = Y_p{v}'*Z.*mu(v);
		[U,~,V] = svd(Temp);
		R{v} = U*V';
    end
	clear Z Temp U V;
    

A = zeros(n,n);
	for v = 1:view_num	
		

        Y_p_spd = Y_p{v};
        A_temp = squareform(pdist(Y_p_spd,'squaredeuclidean'));
		A = A+ A_temp.*mu(v);
    end
    
    
	B = zeros(n,n);
	for v = 1:view_num
		B = B + mu(v).*S_v{v};
	end
	part_M = B - 0.25.*gamma.*A; 
	Temp =part_M - mean(part_M)*eye(n) + ones(n,n)./n;

    for i = 1:n
       
        psi = zeros(n, 1);
		temp = Temp(:,i) + 0.5*mean(psi); 
        
        psi = -2*temp;
        psi(psi<0)=0;
        
		temp = Temp(:,i) + 0.5*mean(psi);  
        
        temp(temp<0)=0;
        S(i,:) = temp;
        
    end
	clear H D V F B A A_temp Y_p_spd part_M Temp psi temp;

	
	temp_obj1 = 0;
	for v = 1:view_num
		temp_obj1 = temp_obj1 + norm(X{v}'*P{v}-Y_p{v},'fro')^2 + alpha*norm(P{v},'fro')^2;
	end
	t_obj2 = 0;
	for v = 1:view_num
		Kesi = 0.5*pdist(Y_p{v},'squaredeuclidean');
		mid = S.*squareform(Kesi);
		t_obj2 = t_obj2 + gamma*mu(v)*sum(mid(:));
	end
	t_obj3 = 0;
	tem_s_sum = zeros(n,n);
	for v = 1:view_num
		tem_s_sum = tem_s_sum + mu(v).*S_v{v};
	end
	t_obj3 = norm(S-tem_s_sum,'fro')^2;
	t_obj4 = 0;
	tem_y_sum = zeros(n,clu_num);
	for v = 1:view_num
		tem_y_sum = tem_y_sum + mu(v)*Y_p{v}*R{v};
	end
	t_obj4 = norm(Y-tem_y_sum,'fro')^2*beta;
	obj(iter) = temp_obj1 + t_obj2 + t_obj3 + t_obj4;

	if iter>1
		if abs((obj(iter)-obj(iter-1))/obj(iter-1))<0.0001
			break;
		end
	end

end

end


