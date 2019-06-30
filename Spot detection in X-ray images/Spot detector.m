clear;clc;
% load training image
fileName="E:\DATAMINING_PROJECT\images_training.tiff" % PLEASE CHANGE THE FILE PATH 
tiffInfo = imfinfo(fileName);  %# Get the TIFF file information
no_frame = numel(tiffInfo);    %# Get the number of images in the file
trainset = cell(no_frame,1);      %# Preallocate the cell array
for iFrame = 1:no_frame
  trainset{iFrame} = double(imread(fileName,'Index',iFrame,'Info',tiffInfo));
end


for i=1:length(trainset)    
    baseFileName = sprintf('Image#%d.tif', i);
    folder='E:\DATAMINING_PROJECT\extract';
    fullFileName = fullfile(folder, baseFileName);
    imwrite(uint8(trainset{i}), fullFileName)
end

%% Load and Explore Image Data

% load images in natural order using function by Stephen Cobeldick
D = 'E:\DATAMINING_PROJECT\extract'; % PLEASE CHANGE THE FILE PATH
S = dir(fullfile(D,'*.tif'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
imds = imageDatastore(F)

%load labels to image datastore
descriptions_training=importdata('E:\DATAMINING_PROJECT\descriptions_training.csv'); % PLEASE CHANGE THE FILE PATH
numberOfImages=length(imds.Files);
labels=descriptions_training(1:numberOfImages,2);
imds.Labels=categorical(labels);


%% Specify Training and Validation Sets
% Divide the data into training and validation data sets, so that each category 
% in the training set contains 30% images, and the validation set contains the 
% remaining images. splitEachLabel splits the datastore digitData 
% into two new datastores, train and validation.

numTrainFiles = .7;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Define Deep Neural Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([24 24 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(2,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

%% Specify Neural Network Training Options
options = trainingOptions('adam', ...
    'ExecutionEnvironment','multi-gpu', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',20, ...
    'Verbose',true, ...
    'Plots','training-progress');


%% Train Neural Network Using Training Data
net = trainNetwork(imdsTrain,layers,options);


%% Classify Validation Images and Compute Accuracy
% Predict the labels of the validation data using the trained network, and calculate 
% the final validation accuracy. Accuracy is the fraction of labels that the network 
% predicts correctly.

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

% display accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

% confusion matrix
confu=confusionmat(YValidation,YPred)


