function SBBoVW_10_3_102_V3()
addpath(genpath('vlfeat2'));
addpath('VLFEATROOT\toolbox\mex\mexw64');
conf.calDir1 ='DataBases\101_ObjectCategories' ;
conf.calDir2 ='DataBases\101_ObjectCategories_seg_1pic' ;
conf.calDir3 ='DataBases\101_ObjectCategories_seg_sal' ;

conf.dataDir = '' ;
conf.numTrain = 10 ;
conf.numTest = 3 ;
conf.numClasses = 102 ;
conf.numWords = 2048 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;

conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = true ;
conf.prefix = 'SBBoVW_10_3_102_V3' ;
conf.randSeed = 1 ;


conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;
classes = dir(conf.calDir1) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;
images = {} ;
imageClass = {} ;
for ci = 1:length(classes)
  ims = dir(fullfile(conf.calDir1, classes{ci}, '*.jpg'))' ;
  ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  images = {images{:}, ims{:}} ;
  imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
selTest = setdiff(1:length(images), selTrain) ;
imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify2 ;
if ~exist(conf.vocabPath) || conf.clobber

  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = vl_colsubset(selTrain, 30) ;
  descrs = {} ;
  for ii = 1:length(selTrainFeats)
  %parfor ii = 1:length(selTrainFeats)
    im = imread(fullfile(conf.calDir1, images{selTrainFeats(ii)})) ;
    
%     image = imread(['images\',D(n).name]);
%     saliencyMap = imread(['saliencyMaps\',D(n).name]);
    mask1=imread(fullfile(conf.calDir3, images{selTrainFeats(ii)}));
      [croped,masknew]=maskcrop(im,mask1)
%     newMatrix = cat(3,segment,segment,segment);
%     im=newMatrix.*image;
   
    
%     im2 = imread(fullfile(conf.calDir2, images{selTrainFeats(ii)})) ;
    im2=croped;
   
    im = im2single(rgb2gray(im)) ;
    im = standarizeImage(im) ; 
    if size(im2,3)>1
    im2 = im2single(rgb2gray(im2)) ;
    else
        im2=im2single(im2);
    end
    im2 = standarizeImage(im2) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
    [drop2, descrs2{ii}] = vl_phow(im2, model.phowOpts{:}) ;
    %drop=
    
    descrs{ii}=cat(2,descrs{ii},descrs2{ii},descrs2{ii});
    %%descrs{ii}=cat(2,descrs2{ii});
    drop=cat(2,drop,drop2,drop2);
% %     
% %     im = im2single(rgb2gray(im)) ;
% %     %im = standarizeImage(im) ;
% %     [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
% %     im2 = imread(fullfile(conf.calDir2, images{selTrainFeats(ii)})) ;
% %     im2 = im2single(rgb2gray(im2)) ;
% %     %im = standarizeImage(im) ;
% %     [drop2, descrs2{ii}] = vl_phow(im2, model.phowOpts{:}) ;
% %     %drop=
% %     descrs{ii}=cat(2,descrs{ii},descrs2{ii},descrs2{ii});
% %     drop=cat(2,drop,drop2,drop2);
  end

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  save(conf.vocabPath, 'vocab') ;
else
  load(conf.vocabPath) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end
%%
if ~exist(conf.histPath) || conf.clobber
  hists = {} ; hists1 = {} ; hists2 = {} ;
  for ii = 1:length(images)
  % for ii = 1:length(images)
    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
% %     im = imread(fullfile(conf.calDir1, images{ii})) ;
% %     im2 = imread(fullfile(conf.calDir2, images{ii})) ;
% %     im2 = im2single(rgb2gray(im2)) ;
% %     
% %     hists{ii} = getImageDescriptor2(model, im,im2);
     im = imread(fullfile(conf.calDir1, images{ii})) ;
     mask1 = imread(fullfile(conf.calDir3, images{ii})) ;
     [croped,masknew]=maskcrop(im,mask1);
     if size(im,3)>1
     im = im2single(rgb2gray(im)) ;
     else
         im = im2single(im);
     end
     im2=croped;
     if size(im2,3)>1
     im2 = im2single(rgb2gray(im2)) ;
     else
         im2=im2single(im2) ;
     end
     hists{ii} = getImageDescriptor2(model, im,im2);
%     hists2{ii} = getImageDescriptor(model,im2);
%     hists{ii}=hists1{ii}+hists2{ii}+hists2{ii};
  end

  hists = cat(2, hists{:}) ;
  save(conf.histPath, 'hists') ;
else
  load(conf.histPath) ;
end
%%

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      parfor ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (imageClass(selTrain) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end
% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)/conf.numTest) )) ;
%print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;




% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;
function hist = getImageDescriptor2(model, im,im2)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
im2 = standarizeImage(im2) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

    [frames2, descrs2] = vl_phow(im2, model.phowOpts{:}) ;
    
    descrs=cat(2,descrs,descrs2,descrs2);
    frames=cat(2,frames,frames2,frames2);
% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;
% -------------------------------------------------------------------------
function [className, score] = classify2(model, im,im2)
% -------------------------------------------------------------------------

hist = getImageDescriptor2(model, im,im2) ;
% hist2 = getImageDescriptor(model,im2) ;
% hist = hist1+hist2+hist2;
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;
