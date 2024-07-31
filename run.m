clc;  close all; clear all;
currentFolder = pwd;
addpath(genpath(currentFolder));
addpath('functions/');
resultdir = 'Results/';
datadir = 'datasets/';  %%%Your datasets dir
if(~exist('Results','file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end
dataname = {'NGs','yaleA_3view','bbcsport','ORL_2515k','WebKB_2views','100leaves','WikipediaArticles','BBC'};
numdata = length(dataname);


retry = 10; 
maxIter = 15;
sigma = 2;
BETA = [0.001,0.01,0.1,1,10,100,1000];  
ALPHA = [0.001,0.01,0.1,1,10,100,1000];
GAMMA = [0.001,0.01,0.1,1,10,100,1000];


for cdata = 1:numdata  
    %% read dataset
    idata = cdata;
    dataf = [datadir, cell2mat(dataname(idata))]

	data = load(dataf);
	X= data.X;
	Y_real = data.Y;


clu_num = max(Y_real);
view_num = length(X);
for v = 1:view_num
	X{v} = X{v}';
    X{v} = NormalizeFea(X{v},0);
end



[~,n] = size(X{1});


filepath = strcat(resultdir,char(dataname(idata)),'_Rerun_result.csv');
fid = fopen(filepath, 'w');
if fid == -1
	error('No such dir,Create Results dir');
end
fprintf(fid, 'alpha,beta,gamma,sigma,ACCmean,ACCstd,NMImean,NMIstd,FscoreMean,FscoreSTD,ARImean,ARIstd\n');
	
count = 0;
for a = 1:length(ALPHA)
	for j = 1:length(BETA)
		for z = 1:length(GAMMA)
            P_list = zeros(retry,0);
            ACC_list = zeros(retry,0);
            NMI_list = zeros(retry,0);
            F_list = zeros(retry,0);
            AR_list = zeros(retry,0);
            RI_list = zeros(retry,0);
            R_list = zeros(retry,0);
            for t = 1:retry
				[P,Y_p,Y,mu,R,S,obj] = MDCAW(X,ALPHA(a),BETA(j),GAMMA(z),sigma,clu_num,maxIter);
				Y_pre = zeros(n,1);
				for i = 1:n
					[~,k] = max(Y(i,:));
					Y_pre(i) = k;
				end

				acc = Accuracy(Y_pre,Y_real);
				[~,nmi,~] = compute_nmi(Y_real,Y_pre);
				[f,p,r] = compute_f(Y_real,Y_pre);
				[AR,RI,MI,HI]=RandIndex(Y_real,Y_pre); 
                
                ACC_list(t) = acc;
                NMI_list(t) = nmi;
                F_list(t) = f;
                AR_list(t) = AR;

				
            end
            ACC_mean = mean(ACC_list);
            ACC_std = std(ACC_list);
            NMI_mean = mean(NMI_list);
            NMI_std = std(NMI_list);
            F_mean = mean(F_list);
            F_std = std(F_list);
            AR_mean = mean(AR_list);
            AR_std = std(AR_list);
			fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ALPHA(a),BETA(j),GAMMA(z),sigma,ACC_mean,ACC_std,NMI_mean,NMI_std,F_mean,F_std,AR_mean,AR_std);

            count = count+1;
			disp(count);

        end
	end
end

end

	