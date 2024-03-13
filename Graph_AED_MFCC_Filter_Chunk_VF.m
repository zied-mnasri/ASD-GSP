clear all
close all

graphType=input('Select a graph type (0: sensor, 1: path, 2: ring, 3: full_connected, 4: Nearest Neighbors, 5: Erdos-Renyi) : ')
useTsneTrain=input('Apply t-SNE for training data (0:No, 1:Yes) : ');
useTsneTest=input('Apply t-SNE for test data (0:No, 1:Yes) : ');
useGraphFilter=input('Use graph filtering (0:No, 1:Yes) : ');
predictionMethod=input('Choose a prediction method (0: Kmeans (Casual!), 1:SVDD/OC-SVM, 2:SVDD/Bin-SVM (Overfitting!), 3: FitcSVM/OC-SVM (Overfitting!), 4:FitcSVM/Bin-SVM)')

%%%%%%%%% Load extracted features and labels %%%%%%%%%%%%%%%%%
framedur=input('Give frame duration (sec): ');
shift=input('Give shift rate (between 0.01 and 0.5): ');
load (strcat(['C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniLR\Data\Matrix_signal_frame_Mel-log-features_',num2str(framedur),'_sec_shift_',num2str(shift),'.mat']));    
load (strcat(['C:\Users\ziedm\Desktop\TempDocs\Research\Projects\UniLR\Data\Matrix_signal_frame_labels_',num2str(framedur),'_sec_shift_',num2str(shift),'.mat']));     

%%%%%%%%%%%%%%%%%% Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=size(mel_log_features);
Features=reshape(mel_log_features,M(1)*M(2)*M(3),M(4)*M(5));
MFCC=mel_log_features(:,:,1,:,:);
MFCC=reshape(MFCC,size(MFCC,1)*size(MFCC,2)*size(MFCC,3),size(MFCC,4)*size(MFCC,5));
DcaseFeatures=MFCC.';

%%%%%%%% Labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Events=reshape(frame_event,M(4)*M(5),1);
Events=Events.';
Labels(Events<0)=0;
Labels(Events>=0)=1;
numfiles=size(Labels,2);

%%%%%%%% Graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=16000;
T=64;
frameLength=T;
frameShift=T/4;
NFFT=2^nextpow2(T);
normalize = 1/sqrt(NFFT);
N=size(DcaseFeatures,2);

numfiles2=input(['Give the n. of audio files (max ',num2str(numfiles),') : '])
indexfiles=(1:numfiles2);
WorkFeatures=DcaseFeatures(indexfiles,:);
WorkLabels=Labels(indexfiles).';

switch graphType
            case 0
                G = gsp_sensor(N);
            case 1
                G = gsp_path(N);
            case 2
                G = gsp_ring(N);
            case 3
                G = gsp_full_connected(N);
            case 4
                G=gsp_nn_graph(WorkFeatures');
            case 5
                G = gsp_erdos_renyi(N,0.05);
end

G = gsp_jtv_graph(G,T,fs);
G.jtv.NFFT=NFFT;
G.jtv.transform='dft';
G1 = gsp_compute_fourier_basis(G);

Nf=6;
g=gsp_design_simple_tf(G, Nf);
order=6;
ga=gsp_approx_filter(G1,g,order);
param.order=order;
figure, gsp_plot_filter(G,ga);

for n=1:numfiles2
        display(['Processing file No ', num2str(n),' of ',num2str(numfiles2)])        
        s=WorkFeatures(n,:);              
        
        if (useGraphFilter==1)
            shat=gsp_filter_evaluate(ga,s');
            shat=abs(gsp_jft(G1,shat));
        else 
            shat=gsp_jft(G1,s');
            shat=abs(shat);
        end
        DcaseChunk(n,:)=reshape(shat,size(shat,1)*size(shat,2),1);
        clear s shat
end

%%%%%%%%%%%%%%%%% Apply t-SNE for training data distribution %%%%%%%%%%%%%%%
if (useTsneTrain==1)
    %Parameters
    no_dims=3;
    initial_dims=30;
    perplexity=30;
    labels=WorkLabels;
    %Apply tSNE
    mappedX=tsne(DcaseChunk,[],no_dims,initial_dims,perplexity);
    figure, gscatter(mappedX(:,1),mappedX(:,3),labels);
end

%%%%%%%%%%%% Prediction methods %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Method n.0-Apply K-means %%%%%%%%%%%%%%%%%%%
if (predictionMethod==0)
    [idx,Cent]=kmeans(DcaseChunk,2);
    Pred=idx-1;
    target=WorkLabels;
    Pred=reshape(Pred,length(Pred),1);
    target=reshape(target,length(target),1);
end

%%%%%%%%%% Method n.1-Apply SVDD for OC-SVM %%%%%%%%%%%%%%%%%%%
if (predictionMethod==1)
    data=DcaseChunk;
    label1=WorkLabels;
    label(label1==0)=1;
    label(label1==1)=-1;
    normalData=data(label==1,:);
    numfilesNormal=size(normalData,1);

    randIdx=randperm(numfiles2);
    
    Trainsize=floor(0.8*numfiles2);
    Testsize=floor(0.2*numfiles2);

    randIdxNormal=randperm(numfilesNormal);
    TrainsizeNormal=floor(0.8*numfilesNormal);
    TestsizeNormal=floor(0.2*numfilesNormal);

    trainData=data(randIdx(1:Trainsize),:,:);
    trainDataNormal=normalData(randIdxNormal(1:TrainsizeNormal),:,:);
    testData=data(randIdx(Trainsize+1:Trainsize+Testsize),:,:);
    trainLabel=label(randIdx(1:Trainsize));
    trainLabelNormal=ones(size(trainDataNormal,1),1);
    trainLabel=trainLabel.';    
    testLabel=label(randIdx(Trainsize+1:Trainsize+Testsize));
    testLabel=testLabel.';
    
    % parameter setting
    kernel = Kernel('type', 'gaussian', 'gamma', 0.04);
    cost = 0.3;
    svddParameter = struct('cost', cost,...
                           'kernelFunc', kernel);

    % creat an SVDD object
    svdd = BaseSVDD(svddParameter);
    % train SVDD model
    svdd.train(trainDataNormal, trainLabelNormal);
    % test SVDD model
    results = svdd.test(testData, testLabel);
    
    target(testLabel==-1)=1;
    target(testLabel==1)=0;
    YPred = results.predictedLabel;
    Pred(YPred==-1)=1;
    Pred(YPred==1)=0;
    Pred=reshape(Pred,length(Pred),1);
    target=reshape(target,length(target),1);
end

%%%%%% Method n.2-SVDD for binary classfication %%%%%
if (predictionMethod==2)
    data=DcaseChunk;
    label1=WorkLabels;
    label(label1==0)=1;
    label(label1==1)=-1;
    randIdx=randperm(numfiles2);
    Trainsize=floor(0.8*numfiles2);
    Testsize=floor(0.2*numfiles2);
    trainData=data(randIdx(1:Trainsize),:,:);
    testData=data(randIdx(Trainsize+1:Trainsize+Testsize),:,:);
    trainLabel=label(randIdx(1:Trainsize));
    trainLabel=trainLabel.';
    testLabel=label(randIdx(Trainsize+1:Trainsize+Testsize));
    testLabel=testLabel.';
    
    % parameter setting
    kernel = Kernel('type', 'gaussian', 'gamma', 0.04);
    cost = 0.3;
    svddParameter = struct('cost', cost,...
                           'kernelFunc', kernel);

    % creat an SVDD object
    svdd = BaseSVDD(svddParameter);
    
    % train SVDD model
    svdd.train(trainData, trainLabel);
    % test SVDD model
    results = svdd.test(testData, testLabel);
    
    target(testLabel==-1)=1;
    target(testLabel==1)=0;
    YPred = results.predictedLabel;
    Pred(YPred==-1)=1;
    Pred(YPred==1)=0;
    Pred=reshape(Pred,length(Pred),1);
    target=reshape(target,length(target),1);
end

%%%%%%%%%% Method n3-Apply fitSVM for OC-SVM %%%%%%%%%%%%%%%%%%%
 if (predictionMethod==3)
    data=DcaseChunk;
    label1=WorkLabels;
    label(label1==0)=1;
    label(label1==1)=-1;
    normalData=data(label==1,:);
    numfilesNormal=size(normalData,1);

    randIdx=randperm(numfiles2);
    Trainsize=floor(0.8*numfiles2);
    Testsize=floor(0.2*numfiles2);

    randIdxNormal=randperm(numfilesNormal);
    TrainsizeNormal=floor(0.8*numfilesNormal);
    TestsizeNormal=floor(0.2*numfilesNormal);

    trainData=data(randIdx(1:Trainsize),:,:);
    trainDataNormal=normalData(randIdxNormal(1:TrainsizeNormal),:,:);
    testData=data(randIdx(Trainsize+1:Trainsize+Testsize),:,:);
    
    trainLabel=label(randIdx(1:Trainsize));
    trainLabelNormal=ones(size(trainDataNormal,1),1);
    trainLabel=trainLabel.';
    
    testLabel=label(randIdx(Trainsize+1:Trainsize+Testsize));
    testLabel=testLabel.';
    target=testLabel;
    
    
    rng(1);
    SVMModel = fitcsvm(trainDataNormal,trainLabelNormal,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.15);
    
    target(testLabel==-1)=1;
    target(testLabel==1)=0;
    svInd = SVMModel.IsSupportVector;
    [YPred,score] = predict(SVMModel,testData);
    Pred(YPred == -1)=1;
    Pred(YPred == 1)=0;
    Pred=reshape(Pred,length(Pred),1);
    target=reshape(target,length(target),1);
 end

%%%%%%%%%%%% Method n.4-Apply fitSVM for Binary-SVM %%%%%%%%%%%%%%%%%%%
if (predictionMethod==4)
    data=DcaseChunk;
    label1=WorkLabels;
    label(label1==0)=1;
    label(label1==1)=-1;
    normalData=data(label==1,:);
    numfilesNormal=size(normalData,1);

    randIdx=randperm(numfiles2);
    Trainsize=floor(0.8*numfiles2);
    Testsize=floor(0.2*numfiles2);

    randIdxNormal=randperm(numfilesNormal);
    TrainsizeNormal=floor(0.8*numfilesNormal);
    TestsizeNormal=floor(0.2*numfilesNormal);

    trainData=data(randIdx(1:Trainsize),:,:);
    trainDataNormal=normalData(randIdxNormal(1:TrainsizeNormal),:,:);
    testData=data(randIdx(Trainsize+1:Trainsize+Testsize),:,:);
    
    trainLabel=label(randIdx(1:Trainsize));
    trainLabelNormal=ones(size(trainDataNormal,1),1);
    trainLabel=trainLabel.';
    
    testLabel=label(randIdx(Trainsize+1:Trainsize+Testsize));
    testLabel=testLabel.';
    target=testLabel;
    rng(1);
    SVMModel = fitcsvm(trainData,trainLabel,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.15);
    
    target(testLabel==-1)=1;
    target(testLabel==1)=0;
    
    svInd = SVMModel.IsSupportVector;
    [YPred,score] = predict(SVMModel,testData);
    Pred(YPred == -1)=1;
    Pred(YPred == 1)=0;
    Pred=reshape(Pred,length(Pred),1);
    target=reshape(target,length(target),1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Calculate evaluation metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[c,CM,ind,per]=confusion(target',Pred');
display(['Confusion matrix = '])
CM

Acc=1-c;
display(['Accuracy = ',num2str(Acc)])

numClasses=max(Labels)-min(Labels)+1;
    for i=1:numClasses
        Prec(i)=CM(i,i)/sum(CM(i,:));
        Rec(i)=CM(i,i)/sum(CM(:,i));
        F1(i)=2*Prec(i)*Rec(i)/(Prec(i)+Rec(i));    
    end

Acc=trace(CM)/length(target);
display('Confusion matrix: ')
display(CM)

[tpr,fpr,thresholds] = roc(target',Pred');
figure
res=plotroc(target',Pred','grid');
grid on
auc=trapz([0,fpr,1],[0,tpr,1]);

Metrics=zeros(5,2)
Metrics(1,:)=[auc,0];
Metrics(2,:)=[Acc,0];
Metrics(3,:)=Prec;
Metrics(4,:)=Rec;
Metrics(5,:)=F1;
display([['AUC      ';'Accuracy ';'Precision';'Recall   ';'F1-score '],num2str(round(Metrics,2))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Apply tSNE for test Data %%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (useTsneTest==1)
    no_dims=3;
    initial_dims=min(length(Pred),30);
    perplexity=30;
    labels=target;%Pred;

    %Apply tSNE
    mappedX=tsne(testData,[],no_dims,initial_dims,perplexity);
    figure, gscatter(mappedX(:,1),mappedX(:,3),labels);
end