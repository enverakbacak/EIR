
clear all;
close all;
clc;


load('./hashCodes/hashCodes_64.mat');
data = hashCodes_64;
load('./hashCodes/features_64.mat');
features = features_64;
load('targets.mat');
targets = targets;
load('filenames.mat');
filenames = filenames;
N = length(filenames);

load('./hashCodes/hashCodes_test_images_64.mat');
data_test = hashCodes_test_images_64;
load('./hashCodes/features_test_images_64.mat');
features_test = features_test_images_64;
load('testimages_labels');
testimages_labels = testimages_labels;



queryIndex = 8;
query_hashCodes = data_test(queryIndex,:); 
query_features  = features_test(queryIndex,:);

q_new = repmat(query_hashCodes,N,1);
dist = xor(data, q_new);
hamming_dist = sum(dist,2);

[~,Retrieved_Items_Index] = sort(hamming_dist,'ascend');

R = 10;              % Pick first 10 retrieved images
Retrieved_Items_AT_R = Retrieved_Items_Index(1:R, :);
Retrieved_Items_AT_R_Features = features(Retrieved_Items_AT_R, :);


euclidian_dist = pdist2(query_features, Retrieved_Items_AT_R_Features)';
decision_matrix = [Retrieved_Items_AT_R euclidian_dist];
Retrieved_Items_AT_R_Ranked = sortrows( decision_matrix , 2 ); 
Retrieved_Items = Retrieved_Items_AT_R_Ranked(:,1);



query_label  = testimages_labels(queryIndex,:); 
Retrieved_Items_AT_R_Ranked_Labels = targets(Retrieved_Items,:);


diff = ismember(Retrieved_Items_AT_R_Ranked_Labels, query_label  , 'rows'); 



num_nz = nnz( diff(:,1) );
s = size(diff(:,1), 1);
    
for j=1:s;
        
    % Cummulative sum of the true-positive elements
    CUMM = cumsum(diff);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); % ???????????                    
       
end

avg_Precision = sum(Precision_AT_K(:,1) .* diff(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
acc = num_nz / s;   % accuracy of the best cluster 

% plot(Recall_AT_K, Precision_AT_K);


