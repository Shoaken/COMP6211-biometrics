%% 
clc
close all
clear


%% import images & initialization
% Set the path to the directory containing the image files
img_dir = 'H:\biometrics\biometrics\processed train';

% Get a list of all the image files in the directory
img_files = dir(fullfile(img_dir, '*.jpg'));

% use 1 file to count number
filename = fullfile(img_dir, img_files(1).name);
img_counter = imread(filename);
count_rows=height(img_counter); %Computes number of rows 
count_columns=width(img_counter); %Computes number of columns 
image_number=length(img_files);

% initialization
img_training_grayscale =cell(1, 88);

% Loop through each image file and read it into MATLAB

for i = 1:image_number
    filename = fullfile(img_dir, img_files(i).name);
    image = imread(filename);
    % transform to HSV space
    img_hsv = rgb2hsv(image);
    
    % filter green mask
    green_mask = img_hsv(:,:,1) > 0.2 & img_hsv(:,:,1) < 0.4 & img_hsv(:,:,2) > 0.2 & img_hsv(:,:,3) > 0.2;
    filtered_img = image;
    filtered_img(repmat(green_mask, [1, 1, 3])) = 0;
    
    % transform to grayscale
    img_gray = rgb2gray(filtered_img);
    img_training_grayscale{i} = img_gray;
end


%% background noise filter

bg_medians= cell(1, image_number);
bg_Gaussians= cell(1, image_number);
windowSize=5;

% Median subtraction
for i = 1:image_number
   bg_medians{i} = medfilt2(img_training_grayscale{i}, [windowSize windowSize]);
end


% Mixture of Gaussians

for i = 1:image_number
   bg_Gaussians{i} = imgaussfilt(img_training_grayscale{i},0.3);

end

figure(1);
subplot(1,3,1);
imshow(img_training_grayscale{1});
title('original image');
subplot(1,3,2);
imshow(bg_medians{1});
title('bg medians');
subplot(1,3,3);
imshow(bg_Gaussians{1},[]);
title('bg Gaussians');

%% Edge detection

image_edges_origin=cell(1, image_number);
image_edges_medians=cell(1, image_number);
image_edges_Gaussians=cell(1, image_number);

for i = 1:image_number
    image_edges_origin{i} = edge(img_training_grayscale{i}, 'Canny',[0.0425, 0.08]);
    image_edges_medians{i} = edge(bg_medians{i}, 'Canny', [0.015, 0.030]);
    image_edges_Gaussians{i} = edge(bg_Gaussians{i}, 'Canny', [0.015, 0.030]);
end
figure(2);
subplot(1,3,1);
imshow(image_edges_origin{1});
title('image edges origin');
subplot(1,3,2);
imshow(image_edges_medians{1});
title('image edges medians');
subplot(1,3,3);
imshow(image_edges_Gaussians{1});
title('image edges Gaussians');



%% image segmentation

se90 = strel('line', 8 ,90);
se0 = strel('line', 8 ,0);
seD = strel('diamond',8);

image_dilate_origin=cell(1, image_number);
image_dilate_medians=cell(1, image_number);
image_dilate_Gaussians=cell(1, image_number);

image_fill_origin=cell(1, image_number);
image_fill_medians=cell(1, image_number);
image_fill_Gaussians=cell(1, image_number);

image_clear_origin=cell(1, image_number);
image_clear_medians=cell(1, image_number);
image_clear_Gaussians=cell(1, image_number);
% image dilate


for i = 1:image_number
    image_dilate_origin{i} = imdilate(image_edges_origin{i},[se90 se0]);
    image_dilate_medians{i} = imdilate(image_edges_medians{i},[se90 se0]);
    image_dilate_Gaussians{i} = imdilate(image_edges_Gaussians{i},[se90 se0]);
end

figure(3);
subplot(1,3,1);
imshow(image_dilate_origin{1});
title('image dilate origin');
subplot(1,3,2);
imshow(image_dilate_medians{1});
title('image dilate medians');
subplot(1,3,3);
imshow(image_dilate_Gaussians{1});
title('image dilate Gaussians');

% image fill
for i = 1:image_number
    image_fill_origin{i} = imfill(image_dilate_origin{i},'holes');
    image_fill_medians{i} = imfill(image_dilate_medians{i},'holes');
    image_fill_Gaussians{i} = imfill(image_dilate_Gaussians{i},'holes');
end

figure(4);
subplot(1,3,1);
imshow(image_fill_origin{1});
title('image fill origin');
subplot(1,3,2);
imshow(image_fill_medians{1});
title('image fill medians');
subplot(1,3,3);
imshow(image_fill_Gaussians{1});
title('image fill Gaussians');

% image border clear

for i = 1:image_number
    image_clear_origin{i} = imerode(image_fill_origin{i},seD);
    image_clear_medians{i} = imerode(image_fill_medians{i},seD);
    image_clear_Gaussians{i} = imerode(image_fill_Gaussians{i},seD);
end


figure(5);

subplot(1,3,1);
imshow(image_clear_origin{51});
title('image clear origin');
subplot(1,3,2);
imshow(image_clear_medians{51});
title('image clear medians');
subplot(1,3,3);
imshow(image_clear_Gaussians{51});
title('image clear Gaussians');


% image segmentation
image_segment_median = cell(1, image_number);
image_segment_Gaussians = cell(1, image_number);

for i = 1:image_number
    % crop images
    image_segment_median{i} = imcrop(image_clear_medians{i},[250 0 900 550]);
    image_segment_Gaussians{i} = imcrop(image_clear_Gaussians{i},[250 0 900 550]);

end

for i = 1:image_number
    figure(6);
    subplot(1,2,1);
    imshow(image_segment_median{i});
    title('image clear medians');
    subplot(1,2,2);
    imshow(image_segment_Gaussians{i});
    title('image clear Gaussians');
end
%% process of silihouette & feature extraction
% further filter noise
silhouette_results_median = cell(1, image_number);
silhouette_results_Gaussians = cell(1, image_number);
seD2 = strel('diamond',13);

for i = 1:image_number
    silhouette_results_median{i} = imopen(image_segment_median{i}, seD2);
    silhouette_results_Gaussians{i} = imopen(image_segment_Gaussians{i}, seD2);
end

% extract area feature 

silhouette_features_median = cell(1, image_number);
silhouette_features_Gaussians = cell(1, image_number);

for i = 1:image_number
    % extraction of area sum
    silhouette_features_median{i} = sum(silhouette_results_median{i},2);
    silhouette_features_Gaussians{i} = sum(silhouette_results_Gaussians{i},2);
end

%extract boundary feature
boundary_results_median = cell(1, image_number);
boundary_results_Gaussians = cell(1, image_number);
for i = 1:image_number   
    boundary_results_median{i} = bwboundaries(silhouette_results_median{i}, 'noholes');
    boundary_results_Gaussians{i} = bwboundaries(silhouette_results_Gaussians{i}, 'noholes');
end

% plot boundary

for i = 1:image_number
    
    
    figure(7);
    subplot(1, 2, 1);
    imshow(img_training_grayscale{i});
    title(['Median subtraction - Boundaries, Image ', num2str(i)]);
    hold on;
    
    for k = 1:length(boundary_results_median{i})
        boundary = boundary_results_median{i}{k};
        plot(boundary(:, 2)+250, boundary(:, 1), 'r', 'LineWidth', 2);
    end
    hold off;
    
    subplot(1, 2, 2);
    imshow(img_training_grayscale{i});
    title(['Mixture of Gaussians - Boundaries, Image ', num2str(i)]);
    hold on;
    
    for k = 1:length(boundary_results_Gaussians{i})
        boundary = boundary_results_Gaussians{i}{k};
        plot(boundary(:, 2)+250, boundary(:, 1), 'g', 'LineWidth', 2);
    end
    hold off;
end

% 2D Silhouette 
unwrapped_results_median = cell(1, image_number);
unwrapped_results_Gaussians = cell(1, image_number);

for i = 1:image_number
    % Median subtraction 
    if i <= length(boundary_results_median) && ~isempty(boundary_results_median{i})
        buffer = zeros(1,length(boundary_results_median{i}));
        for j=1:length(boundary_results_median{i})
            buffer(j) = length(boundary_results_median{i}{j});
        end
        [~,detector]= max(buffer);
        boundary_median = boundary_results_median{i}{detector};
        [~, start_idx] = min(boundary_median(:, 1)); % 寻找轮廓的最高点作为起始点
        unwrapped_boundary_median = unwrap_contour(boundary_median, start_idx);
        unwrapped_results_median{i} = unwrapped_boundary_median;
    end
    
    % Mixture of Gaussians 
    if i <= length(boundary_results_Gaussians) && ~isempty(boundary_results_Gaussians{i})
        buffer = zeros(1,length(boundary_results_Gaussians{i}));
        for j=1:length(boundary_results_Gaussians{i})
            buffer(j) = length(boundary_results_Gaussians{i}{j});
        end
        [~,detector]= max(buffer);
        boundary_Gaussians = boundary_results_Gaussians{i}{1};
        [~, start_idx] = min(boundary_Gaussians(:, 1)); % 寻找轮廓的最高点作为起始点
        unwrapped_boundary_Gaussians = unwrap_contour(boundary_Gaussians, start_idx);
        unwrapped_results_Gaussians{i} = unwrapped_boundary_Gaussians;
    end
end


for i = 1:image_number
    figure(8);
    % Median subtraction 结果
    if i <= length(unwrapped_results_median) && ~isempty(unwrapped_results_median{i})
        subplot(1, 2, 1);
        plot(unwrapped_results_median{i}(:, 1), unwrapped_results_median{i}(:, 2), 'r');
        x_min = 0;
        x_max = max(unwrapped_results_median{i}(:, 1));
        if x_min == x_max
            x_max = x_max + 1;
        end
        xlim([x_min x_max]);

        ylim([0 max(unwrapped_results_median{i}(:, 2))]);
        title(['Median subtraction - Unwrapped Contour, Image ', num2str(i)]);
    end
    
    % Mixture of Gaussians 
    if i <= length(unwrapped_results_Gaussians) && ~isempty(unwrapped_results_Gaussians{i})
        subplot(1, 2, 2);
        plot(unwrapped_results_Gaussians{i}(:, 1), unwrapped_results_Gaussians{i}(:, 2), 'b');
        x_min = 0;
        x_max = max(unwrapped_results_Gaussians{i}(:, 1));
        if x_min == x_max
        x_max = x_max + 1;
        end
        xlim([x_min x_max]);

        ylim([0 max(unwrapped_results_Gaussians{i}(:, 2))]);
        title(['Mixture of Gaussians - Unwrapped Contour, Image ', num2str(i)]);
    end
end
%%
% normalization
normalized_results_median = cell(1, image_number);
normalized_results_Gaussians = cell(1, image_number);

for i = 1:image_number
    % Median subtraction 
    if i <= length(unwrapped_results_median) && ~isempty(unwrapped_results_median{i})
        min_val = min(unwrapped_results_median{i}, [], 1);
        max_val = max(unwrapped_results_median{i}, [], 1);
        normalized_results_median{i} = (unwrapped_results_median{i} - min_val) ./ (max_val - min_val);
    end
    
    % Mixture of Gaussians 
    if i <= length(unwrapped_results_Gaussians) && ~isempty(unwrapped_results_Gaussians{i})
        min_val = min(unwrapped_results_Gaussians{i}, [], 1);
        max_val = max(unwrapped_results_Gaussians{i}, [], 1);
        normalized_results_Gaussians{i} = (unwrapped_results_Gaussians{i} - min_val) ./ (max_val - min_val);
    end
end

% normalize result
for i = 1:image_number
    figure(9);
    % Median subtraction 
    if i <= length(normalized_results_median) && ~isempty(normalized_results_median{i})
        subplot(1, 2, 1);
        plot(normalized_results_median{i}(:, 1), normalized_results_median{i}(:, 2), 'r');
        xlim([0 1]);
        ylim([0 1]);
        title(['Median subtraction - Normalized Contour, Image ', num2str(i)]);
    end
    
    % Mixture of Gaussians 
    if i <= length(normalized_results_Gaussians) && ~isempty(normalized_results_Gaussians{i})
        subplot(1, 2, 2);
        plot(normalized_results_Gaussians{i}(:, 1), normalized_results_Gaussians{i}(:, 2), 'b');
        xlim([0 1]);
        ylim([0 1]);
        title(['Mixture of Gaussians - Normalized Contour, Image ', num2str(i)]);
    end
end

%% Feature analysis

%% Elliptic Fourier Descriptors
% 1. Obtain a complex function from the 2D curve
complex_results_median = cell(1, image_number);
complex_results_Gaussians = cell(1, image_number);

for i = 1:image_number
    % Median subtraction 结果
    if i <= length(normalized_results_median) && ~isempty(normalized_results_median{i})
        complex_results_median{i} = normalized_results_median{i}(:, 2) + 1j * normalized_results_median{i}(:, 1);
    end

    % Mixture of Gaussians 结果
    if i <= length(normalized_results_Gaussians) && ~isempty(normalized_results_Gaussians{i})
        complex_results_Gaussians{i} = normalized_results_Gaussians{i}(:, 2) + 1j * normalized_results_Gaussians{i}(:, 1);
    end
end

% Visualize complex functions (Real and Imaginary parts separately)
for i = 1:image_number
    figure(10);
    subplot(2, 2, 1);
    plot(real(complex_results_median{i}), 'r');
    title(['Median subtraction - Real part, Image ', num2str(i)]);
    subplot(2, 2, 2);
    plot(real(complex_results_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Real part, Image ', num2str(i)]);
    subplot(2, 2, 3);
    plot(imag(complex_results_median{i}), 'r');
    title(['Median subtraction - Imaginary part, Image ', num2str(i)]);
    subplot(2, 2, 4);
    plot(imag(complex_results_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Imaginary part, Image ', num2str(i)]);
end


% 2. Perform a Fourier Trigonometric expansion

n = 30; 
fourier_coeffs_median = cell(1, image_number);
fourier_coeffs_Gaussians = cell(1, image_number);
% Calculate Fourier coefficients

for i = 1:image_number
    % Median subtraction 

    if i <= length(complex_results_median) && ~isempty(complex_results_median{i})
        fourier_coeffs_median{i} = fft(complex_results_median{i}, n);
    end
    
    % Mixture of Gaussians 
    if i <= length(complex_results_Gaussians) && ~isempty(complex_results_Gaussians{i})
        fourier_coeffs_Gaussians{i} = fft(complex_results_Gaussians{i}, n);
    end
end

% Visualize Fourier coefficients
for i = 1:image_number
    figure(11);
    subplot(1, 2, 1);
    stem(abs(fourier_coeffs_median{i}), 'r');
    title(['Median subtraction - Fourier coefficients, Image ', num2str(i)]);
    subplot(1, 2, 2);
    stem(abs(fourier_coeffs_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Fourier coefficients, Image ', num2str(i)]);
end

% 3. Define descriptors from Fourier coefficients using Translation Invariant Fourier Descriptors (TIFD)

invariant_descriptors_median = cell(1, image_number);
invariant_descriptors_Gaussians = cell(1, image_number);

for i = 1:image_number

    % Median subtraction 结果
    if i <= length(fourier_coeffs_median) && ~isempty(fourier_coeffs_median{i})
        tifd = fourier_coeffs_median{i}(2:end) ./ fourier_coeffs_median{i}(1); % 计算 TIFD
        invariant_descriptors_median{i} = abs(tifd); % 取绝对值作为平移不变描述子
    end

    % Mixture of Gaussians 结果

    if i <= length(fourier_coeffs_Gaussians) && ~isempty(fourier_coeffs_Gaussians{i})
        tifd = fourier_coeffs_Gaussians{i}(2:end) ./ fourier_coeffs_Gaussians{i}(1); % 计算 TIFD
        invariant_descriptors_Gaussians{i} = abs(tifd); % 取绝对值作为平移不变描述子
    end
end


% Visualize Invariant Descriptors (TIFD)

for i = 1:image_number

    figure(12);

    subplot(1, 2, 1);

    stem(invariant_descriptors_median{i}, 'r');

    title(['Median subtraction - TIFD, Image ', num2str(i)]);

    subplot(1, 2, 2);

    stem(invariant_descriptors_Gaussians{i}, 'g');

    title(['Mixture of Gaussians - TIFD, Image ', num2str(i)]);

end

%%

% 计算欧几里得距离矩阵

num_descriptors = length(invariant_descriptors_median);
distance_matrix_median = zeros(num_descriptors, num_descriptors);
distance_matrix_Gaussians = zeros(num_descriptors, num_descriptors);


for i = 1:num_descriptors
    for j = 1:num_descriptors
        distance_matrix_median(i, j) = norm(invariant_descriptors_median{i} - invariant_descriptors_median{j});
        distance_matrix_Gaussians(i, j) = norm(invariant_descriptors_Gaussians{i} - invariant_descriptors_Gaussians{j});
    end
end



% 可视化欧几里得距离矩阵

figure(13);
subplot(1, 2, 1);
imagesc(distance_matrix_median);
title('Median subtraction - Euclidean distance matrix');
colorbar;
axis square;


subplot(1, 2, 2);
imagesc(distance_matrix_Gaussians);
title('Mixture of Gaussians - Euclidean distance matrix');
colorbar;
axis square;

% 散点图

figure(14);
subplot(1, 2, 1);
scatter(distance_matrix_median(:), 1:numel(distance_matrix_median));
title('Median subtraction - Euclidean distances');
xlabel('Pairwise distance index');
ylabel('Euclidean distance');


subplot(1, 2, 2);
scatter(distance_matrix_Gaussians(:), 1:numel(distance_matrix_Gaussians));
title('Mixture of Gaussians - Euclidean distances');
xlabel('Pairwise distance index');
ylabel('Euclidean distance');


% 直方图

figure(15);
subplot(1, 2, 1);
hist(distance_matrix_median(:), 20); % 您可以更改直方图的柱子数量
title('Median subtraction - Euclidean distance histogram');
xlabel('Euclidean distance');
ylabel('Frequency');


subplot(1, 2, 2);
hist(distance_matrix_Gaussians(:), 20); % 您可以更改直方图的柱子数量
title('Mixture of Gaussians - Euclidean distance histogram');
xlabel('Euclidean distance');
ylabel('Frequency');


% 输出 Median subtraction 欧几里得距离矩阵

disp('Median subtraction - Euclidean distance matrix:');

for i = 1:num_descriptors
    for j = 1:num_descriptors
        fprintf('%0.4f\t', distance_matrix_median(i, j));
    end
    fprintf('\n');
end

% 输出 Mixture of Gaussians 欧几里得距离矩阵

disp('Mixture of Gaussians - Euclidean distance matrix:');

for i = 1:num_descriptors
    for j = 1:num_descriptors
        fprintf('%0.4f\t', distance_matrix_Gaussians(i, j));
    end
    fprintf('\n');
end




feature_space_median = cell(4,image_number);
feature_space_Gaussians = cell(4,image_number);

for i = 1:image_number
    feature_space_median{1,i} = silhouette_features_median{i}.';
    feature_space_Gaussians{1,i} = silhouette_features_Gaussians{i}.';
    feature_space_median{2,i} = distance_matrix_median(i,:);
    feature_space_Gaussians{2,i} = distance_matrix_Gaussians(i,:);
%     feature_space_median{2,i} = invariant_descriptors_median{i}.';
%     feature_space_Gaussians{2,i} = invariant_descriptors_Gaussians{i}.';
    feature_space_median{3,i} = 1:n;
    feature_space_Gaussians{3,i} = 1:n;
    feature_space_median{4,i} = extractBefore(img_files(i).name,8);
    feature_space_Gaussians{4,i} = extractBefore(img_files(i).name,8);
end





%% Import test images


% Train Data 

img_test_dir = 'H:\biometrics\biometrics\processed test';

% Get a list of all the image files in the directory
img_test_files = dir(fullfile(img_test_dir, '*.jpg'));
image_test_number=length(img_test_files);
img_test_grayscale =cell(1, image_test_number);


% import image and filter
for i = 1:image_test_number
    filename = fullfile(img_test_dir, img_test_files(i).name);
    image = imread(filename);
    % transform to HSV space
    img_hsv = rgb2hsv(image);
    
    % filter green mask
    green_mask = img_hsv(:,:,1) > 0.2 & img_hsv(:,:,1) < 0.4 & img_hsv(:,:,2) > 0.2 & img_hsv(:,:,3) > 0.2;
    filtered_img = image;
    filtered_img(repmat(green_mask, [1, 1, 3])) = 0;
    
    % transform to grayscale
    img_gray = rgb2gray(filtered_img);
    img_test_grayscale{i} = img_gray;
end

bg_test_medians= cell(1, image_test_number);
bg_test_Gaussians= cell(1, image_test_number);
windowSize=5;

% Median subtraction
for i = 1:image_test_number
   bg_test_medians{i} = medfilt2(img_test_grayscale{i}, [windowSize windowSize]);
end



% Mixture of Gaussians


for i = 1:image_test_number
   bg_test_Gaussians{i} = imgaussfilt(img_test_grayscale{i},0.8);

end

figure(16);
subplot(1,3,1);
imshow(img_test_grayscale{1});
title('original test image');
subplot(1,3,2);
imshow(bg_test_medians{1});
title('bg test medians');
subplot(1,3,3);
imshow(bg_test_Gaussians{1},[]);
title('bg test Gaussians');

test_edges_origin=cell(1, image_test_number);
test_edges_medians=cell(1, image_test_number);
test_edges_Gaussians=cell(1, image_test_number);

for i = 1:image_test_number
    test_edges_origin{i} = edge(img_test_grayscale{i}, 'Canny',[0.0425, 0.08]);
    test_edges_medians{i} = edge(bg_test_medians{i}, 'Canny', [0.015, 0.030]);
    test_edges_Gaussians{i} = edge(bg_test_Gaussians{i}, 'Canny', [0.015, 0.030]);
end
figure(17);
subplot(1,3,1);
imshow(test_edges_origin{1});
title('test edges origin');
subplot(1,3,2);
imshow(test_edges_medians{1});
title('test edges medians');
subplot(1,3,3);
imshow(test_edges_Gaussians{1});
title('test edges Gaussians');

% image segmentation
test_dilate_origin=cell(1, image_test_number);
test_dilate_medians=cell(1, image_test_number);
test_dilate_Gaussians=cell(1, image_test_number);

test_fill_origin=cell(1, image_test_number);
test_fill_medians=cell(1, image_test_number);
test_fill_Gaussians=cell(1, image_test_number);

test_clear_origin=cell(1, image_test_number);
test_clear_medians=cell(1, image_test_number);
test_clear_Gaussians=cell(1, image_test_number);

se90_test = strel('line', 8 ,90);
se0_test = strel('line', 8 ,0);
seD_test = strel('diamond',8);
seD2_test = strel('diamond',13);

% image dilate


for i = 1:image_test_number
    test_dilate_origin{i} = imdilate(test_edges_origin{i},[se90_test se0_test]);
    test_dilate_medians{i} = imdilate(test_edges_medians{i},[se90_test se0_test]);
    test_dilate_Gaussians{i} = imdilate(test_edges_Gaussians{i},[se90_test se0_test]);
end

figure(18);
subplot(1,3,1);
imshow(test_dilate_origin{1});
title('test dilate origin');
subplot(1,3,2);
imshow(test_dilate_medians{1});
title('test dilate medians');
subplot(1,3,3);
imshow(test_dilate_Gaussians{1});
title('test dilate Gaussians');

% image fill
for i = 1:image_test_number
    test_fill_origin{i} = imfill(test_dilate_origin{i},'holes');
    test_fill_medians{i} = imfill(test_dilate_medians{i},'holes');
    test_fill_Gaussians{i} = imfill(test_dilate_Gaussians{i},'holes');
end

figure(19);
subplot(1,3,1);
imshow(test_fill_origin{1});
title('test fill origin');
subplot(1,3,2);
imshow(test_fill_medians{1});
title('test fill medians');
subplot(1,3,3);
imshow(test_fill_Gaussians{1});
title('test fill Gaussians');

% image border clear

for i = 1:image_test_number
    test_clear_origin{i} = imerode(test_fill_origin{i},seD_test);
    test_clear_medians{i} = imerode(test_fill_medians{i},seD_test);
    test_clear_Gaussians{i} = imerode(test_fill_Gaussians{i},seD_test);
end


figure(20);
subplot(1,3,1);
imshow(test_clear_origin{1});
title('test clear origin');
subplot(1,3,2);
imshow(test_clear_medians{1});
title('test clear medians');
subplot(1,3,3);
imshow(test_clear_Gaussians{1});
title('test clear Gaussians');


% image segmentation
test_segment_median = cell(1, image_test_number);
test_segment_Gaussians = cell(1, image_test_number);
figure(21);
for i = 1:image_test_number
    % crop images
    test_segment_median{i} = imcrop(test_clear_medians{i},[250 0 900 550]);
    imshow(test_segment_median{i});
    test_segment_Gaussians{i} = imcrop(test_clear_Gaussians{i},[250 0 900 550]);
    imshow(test_segment_Gaussians{i});

end

% feature extraction
silhouette_test_median = cell(1, image_test_number);
silhouette_test_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number
    silhouette_test_median{i} = imopen(test_segment_median{i}, seD2_test);
    silhouette_test_Gaussians{i} = imopen(test_segment_Gaussians{i}, seD2_test);
end

% extract area feature 

silhouette_test_features_median = cell(1, image_test_number);
silhouette_test_features_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number
    % extraction of area sum
    silhouette_test_features_median{i} = sum(silhouette_test_median{i},2);
    silhouette_test_features_Gaussians{i} = sum(silhouette_test_Gaussians{i},2);
    
end

%extract boundary feature
boundary_test_median = cell(1, image_test_number);
boundary_test_Gaussians = cell(1, image_test_number);
for i = 1:image_test_number
    boundary_test_median{i} = bwboundaries(silhouette_test_median{i}, 'noholes');
    boundary_test_Gaussians{i} = bwboundaries(silhouette_test_median{i}, 'noholes');
end

% plot boundary

for i = 1:image_test_number
        
    figure(22);
    subplot(1, 2, 1);
    imshow(img_test_grayscale{i});
    title(['Median subtraction - Boundaries, Image ', num2str(i)]);
    hold on;
    
    for k = 1:length(boundary_test_median{i})
        boundary = boundary_test_median{i}{k};
        plot(boundary(:, 2)+250, boundary(:, 1), 'r', 'LineWidth', 2);
    end
    hold off;
    
    subplot(1, 2, 2);
    imshow(img_test_grayscale{i});
    title(['Mixture of Gaussians - Boundaries, Image ', num2str(i)]);
    hold on;
    
    for k = 1:length(boundary_test_Gaussians{i})
        boundary = boundary_test_Gaussians{i}{k};
        plot(boundary(:, 2)+250, boundary(:, 1), 'g', 'LineWidth', 2);
    end
    hold off;
end

% 2D Silhouette 
unwrapped_test_median = cell(1, image_test_number);
unwrapped_test_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number
    % Median subtraction 
    if i <= length(boundary_test_median) && ~isempty(boundary_test_median{i})
        buffer = zeros(1,length(boundary_test_median{i}));
        for j=1:length(boundary_test_median{i})
            buffer(j) = length(boundary_test_median{i}{j});
        end
        [~,detector]= max(buffer);
        boundary_median = boundary_test_median{i}{detector};
        [~, start_idx] = min(boundary_median(:, 1)); % 寻找轮廓的最高点作为起始点
        unwrapped_boundary_median = unwrap_contour(boundary_median, start_idx);
        unwrapped_test_median{i} = unwrapped_boundary_median;
    end
    
    % Mixture of Gaussians 
    if i <= length(boundary_test_Gaussians) && ~isempty(boundary_test_Gaussians{i})
        buffer = zeros(1,length(boundary_test_Gaussians{i}));
        for j=1:length(boundary_test_Gaussians{i})
            buffer(j) = length(boundary_test_Gaussians{i}{j});
        end
        [~,detector]= max(buffer);
        boundary_Gaussians = boundary_test_Gaussians{i}{detector};
        [~, start_idx] = min(boundary_Gaussians(:, 1)); % 寻找轮廓的最高点作为起始点
        unwrapped_boundary_Gaussians = unwrap_contour(boundary_Gaussians, start_idx);
        unwrapped_test_Gaussians{i} = unwrapped_boundary_Gaussians;
    end
end


for i = 1:image_test_number
    figure(23);
    % Median subtraction 结果
    if i <= length(unwrapped_test_median) && ~isempty(unwrapped_test_median{i})
        subplot(1, 2, 1);
        plot(unwrapped_test_median{i}(:, 1), unwrapped_test_median{i}(:, 2), 'r');
        %xlim([0 max(unwrapped_results_median{i}(:, 1))]);
        x_min = 0;
        x_max = max(unwrapped_test_median{i}(:, 1));
        if x_min == x_max
            x_max = x_max + 1;
        end
        xlim([x_min x_max]);

        ylim([0 max(unwrapped_test_median{i}(:, 2))]);
        title(['Median subtraction - Unwrapped Contour, Image ', num2str(i)]);
    end
    
    % Mixture of Gaussians 
    if i <= length(unwrapped_test_Gaussians) && ~isempty(unwrapped_test_Gaussians{i})
        subplot(1, 2, 2);
        plot(unwrapped_test_Gaussians{i}(:, 1), unwrapped_test_Gaussians{i}(:, 2), 'b');
        %xlim([0 max(unwrapped_results_Gaussians{i}(:, 1))]);
        x_min = 0;
        x_max = max(unwrapped_test_Gaussians{i}(:, 1));
        if x_min == x_max
        x_max = x_max + 1;
        end
        xlim([x_min x_max]);

        ylim([0 max(unwrapped_test_Gaussians{i}(:, 2))]);
        title(['Mixture of Gaussians - Unwrapped Contour, Image ', num2str(i)]);
    end
end
%%
% normalization
normalized_test_median = cell(1, image_test_number);
normalized_test_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number
    % Median subtraction 
    if i <= length(unwrapped_test_median) && ~isempty(unwrapped_test_median{i})
        min_val = min(unwrapped_test_median{i}, [], 1);
        max_val = max(unwrapped_test_median{i}, [], 1);
        normalized_test_median{i} = (unwrapped_test_median{i} - min_val) ./ (max_val - min_val);
    end
    
    % Mixture of Gaussians 
    if i <= length(unwrapped_test_Gaussians) && ~isempty(unwrapped_test_Gaussians{i})
        min_val = min(unwrapped_test_Gaussians{i}, [], 1);
        max_val = max(unwrapped_test_Gaussians{i}, [], 1);
        normalized_test_Gaussians{i} = (unwrapped_test_Gaussians{i} - min_val) ./ (max_val - min_val);
    end
end

% normalize result
for i = 1:image_test_number
    figure(24);
    % Median subtraction 
    if i <= length(normalized_test_median) && ~isempty(normalized_test_median{i})
        subplot(1, 2, 1);
        plot(normalized_test_median{i}(:, 1), normalized_test_median{i}(:, 2), 'r');
        xlim([0 1]);
        ylim([0 1]);
        title(['Median subtraction - Normalized Contour, Image ', num2str(i)]);
    end
    
    % Mixture of Gaussians 
    if i <= length(normalized_test_Gaussians) && ~isempty(normalized_test_Gaussians{i})
        subplot(1, 2, 2);
        plot(normalized_test_Gaussians{i}(:, 1), normalized_test_Gaussians{i}(:, 2), 'b');
        xlim([0 1]);
        ylim([0 1]);
        title(['Mixture of Gaussians - Normalized Contour, Image ', num2str(i)]);
    end
end




% Feature analysis


% 1. Obtain a complex function from the 2D curve
complex_test_median = cell(1, image_test_number);
complex_test_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number
    % Median subtraction 
    if i <= length(normalized_test_median) && ~isempty(normalized_test_median{i})
        complex_test_median{i} = normalized_test_median{i}(:, 2) + 1j * normalized_test_median{i}(:, 1);
%         complex_test_median{i} = (boundary_test_median{i}{detector}(:, 2)-central_x_median) + 1j * (boundary_test_median{i}{detector}(:, 1)-central_y_median);
    end
    
    % Mixture of Gaussians 
    if i <= length(normalized_test_Gaussians) && ~isempty(normalized_test_Gaussians{i})
        complex_test_Gaussians{i} = normalized_test_Gaussians{i}(:, 2) + 1j * normalized_test_Gaussians{i}(:, 1);
%         complex_test_Gaussians{i} = (boundary_test_Gaussians{i}{detector}(:, 2)-central_x_Gaussians) + 1j * (boundary_test_Gaussians{i}{detector}(:, 1)-central_y_Gaussians);
    end
end

% Visualize complex functions (Real and Imaginary parts separately)
for i = 1:image_test_number
    figure(25);
    subplot(2, 2, 1);
    plot(real(complex_test_median{i}), 'r');
    title(['Median subtraction - Real part, Image ', num2str(i)]);
    subplot(2, 2, 2);
    plot(real(complex_test_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Real part, Image ', num2str(i)]);
    subplot(2, 2, 3);
    plot(imag(complex_test_median{i}), 'r');
    title(['Median subtraction - Imaginary part, Image ', num2str(i)]);
    subplot(2, 2, 4);
    plot(imag(complex_test_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Imaginary part, Image ', num2str(i)]);
end


% 2. Perform a Fourier Trigonometric expansion

n = 30; 
fourier_coeffs_test_median = cell(1, image_test_number);
fourier_coeffs_test_Gaussians = cell(1, image_test_number);

% Calculate Fourier coefficients
for i = 1:image_test_number
    % Median subtraction 

    if i <= length(complex_test_median) && ~isempty(complex_test_median{i})
        fourier_coeffs_test_median{i} = fft(complex_test_median{i}, n);
    end
    
    % Mixture of Gaussians 
    if i <= length(complex_test_Gaussians) && ~isempty(complex_test_Gaussians{i})
        fourier_coeffs_test_Gaussians{i} = fft(complex_test_Gaussians{i}, n);
    end
end

% Visualize Fourier coefficients
for i = 1:image_test_number
    figure(26);
    subplot(1, 2, 1);
    stem(abs(fourier_coeffs_test_median{i}), 'r');
    title(['Median subtraction - Fourier coefficients, Image ', num2str(i)]);
    subplot(1, 2, 2);
    stem(abs(fourier_coeffs_test_Gaussians{i}), 'g');
    title(['Mixture of Gaussians - Fourier coefficients, Image ', num2str(i)]);
end

% 3. Define descriptors from Fourier coefficients using Translation Invariant Fourier Descriptors (TIFD)

invariant_descriptors_test_median = cell(1, image_test_number);
invariant_descriptors_test_Gaussians = cell(1, image_test_number);

for i = 1:image_test_number

    % Median subtraction 结果
    if i <= length(fourier_coeffs_test_median) && ~isempty(fourier_coeffs_test_median{i})
        tifd = fourier_coeffs_test_median{i}(2:end) ./ fourier_coeffs_test_median{i}(1); % 计算 TIFD
        invariant_descriptors_test_median{i} = abs(tifd); % 取绝对值作为平移不变描述子
    end

    % Mixture of Gaussians 结果

    if i <= length(fourier_coeffs_test_Gaussians) && ~isempty(fourier_coeffs_test_Gaussians{i})
        tifd = fourier_coeffs_test_Gaussians{i}(2:end) ./ fourier_coeffs_test_Gaussians{i}(1); % 计算 TIFD
        invariant_descriptors_test_Gaussians{i} = abs(tifd); % 取绝对值作为平移不变描述子
    end
end


% Visualize Invariant Descriptors (TIFD)

for i = 1:image_test_number

    figure(27);
    subplot(1, 2, 1);
    stem(invariant_descriptors_test_median{i}, 'r');
    title(['Median subtraction - TIFD, Image ', num2str(i)]);
    subplot(1, 2, 2);
    stem(invariant_descriptors_test_Gaussians{i}, 'g');
    title(['Mixture of Gaussians - TIFD, Image ', num2str(i)]);

end

%%

% 计算欧几里得距离矩阵

num_descriptors_test = length(invariant_descriptors_test_median);
distance_matrix_test_median = zeros(num_descriptors_test, num_descriptors);
distance_matrix_test_Gaussians = zeros(num_descriptors_test, num_descriptors);

for i = 1:num_descriptors_test
    for j = 1:num_descriptors
        distance_matrix_test_median(i, j) = norm(invariant_descriptors_test_median{i} - invariant_descriptors_median{j});
        distance_matrix_test_Gaussians(i, j) = norm(invariant_descriptors_test_Gaussians{i} - invariant_descriptors_Gaussians{j});
    end
end



% 可视化欧几里得距离矩阵

figure(28);
subplot(1, 2, 1);
imagesc(distance_matrix_test_median);
title('Median subtraction - Euclidean distance matrix');
colorbar;
axis square;


subplot(1, 2, 2);
imagesc(distance_matrix_test_Gaussians);
title('Mixture of Gaussians - Euclidean distance matrix');
colorbar;
axis square;

% 散点图

figure(29);
subplot(1, 2, 1);
scatter(distance_matrix_test_median(:), 1:numel(distance_matrix_test_median));
title('Median subtraction - Euclidean distances');
xlabel('Pairwise distance index');
ylabel('Euclidean distance');


subplot(1, 2, 2);
scatter(distance_matrix_test_Gaussians(:), 1:numel(distance_matrix_test_Gaussians));
title('Mixture of Gaussians - Euclidean distances');
xlabel('Pairwise distance index');
ylabel('Euclidean distance');


% 直方图

figure(30);
subplot(1, 2, 1);
hist(distance_matrix_test_median(:), 20); % 您可以更改直方图的柱子数量
title('Median subtraction - Euclidean distance histogram');
xlabel('Euclidean distance');
ylabel('Frequency');


subplot(1, 2, 2);
hist(distance_matrix_test_Gaussians(:), 20); % 您可以更改直方图的柱子数量
title('Mixture of Gaussians - Euclidean distance histogram');
xlabel('Euclidean distance');
ylabel('Frequency');


% 输出 Median subtraction 欧几里得距离矩阵

disp('Median subtraction - Euclidean distance matrix:');

for i = 1:num_descriptors_test
    for j = 1:num_descriptors
        fprintf('%0.4f\t', distance_matrix_test_median(i, j));
    end
    fprintf('\n');
end

% 输出 Mixture of Gaussians 欧几里得距离矩阵

disp('Mixture of Gaussians - Euclidean distance matrix:');

for i = 1:num_descriptors_test
    for j = 1:num_descriptors
        fprintf('%0.4f\t', distance_matrix_test_Gaussians(i, j));
    end
    fprintf('\n');
end

feature_space_test_median = cell(4,image_test_number);
feature_space_test_Gaussians = cell(4,image_test_number);

for i = 1:image_test_number
    feature_space_test_median{1,i} = silhouette_test_features_median{i}.';
    feature_space_test_Gaussians{1,i} = silhouette_test_features_Gaussians{i}.';
    feature_space_test_median{2,i} = distance_matrix_test_median(i,:);
    feature_space_test_Gaussians{2,i} = distance_matrix_test_Gaussians(i,:);
%     feature_space_test_median{2,i} = invariant_descriptors_test_median{i}.';
%     feature_space_test_Gaussians{2,i} = invariant_descriptors_test_Gaussians{i}.';
    feature_space_test_median{3,i} = 1:n;
    feature_space_test_Gaussians{3,i} = 1:n;
    feature_space_test_median{4,i} = extractBefore(img_test_files(i).name,9);
    feature_space_test_Gaussians{4,i} = extractBefore(img_test_files(i).name,9);
end


%% Classification

% Set up the training data
feature_space = cell2table(feature_space_median'); % Choose either feature_space_median or feature_space_Gaussians

X = [feature_space.Var1,feature_space.Var2,feature_space.Var3];
% X = [feature_space.Var2,feature_space.Var3];
Y= [feature_space.Var4];

% Train a KNN model

KNNModel = fitcknn(X,Y,'DistanceWeight','inverse');

% Evaluate the model on the validation set
[Ypred, ~] = predict(KNNModel, X);
accuracy_counter=0;
for i=1:length(Ypred)
    if Ypred{i}==Y{i}
        accuracy_counter=accuracy_counter+1;
    else
        continue;
    end
end
accuracy = accuracy_counter/length(Ypred);
fprintf('Validation accuracy for training group: %0.2f%%\n', accuracy*100);

% Classify test image
Test_feature_space = cell2table(feature_space_test_Gaussians');
X_test = [Test_feature_space.Var1,Test_feature_space.Var2,Test_feature_space.Var3];
[Ypred_test, Classification_scores] = predict(KNNModel, X_test);

for i=1:length(Ypred_test)
    test_string = string(Ypred_test{i});
    position_idx = find(strcmp(Y, test_string));
    figure(i+30);
    filename = fullfile(img_dir, img_files(position_idx(1)).name);
    image = imread(filename);
    subplot(1,2,1);
    imshow(image);
    title('Front view of matching subject in training set');
    subplot(1,2,2);
    filename = fullfile(img_test_dir, img_test_files(i).name);
    image_test = imread(filename);
    imshow(image_test);
    title('Testing subject');
end
correct_result=cell(length(Ypred_test),1);
for i=1:2:length(Ypred_test)
    correct_result{i}=Ypred_test{i};
    correct_result{i+1}=correct_result{i};
end
accuracy_counter=0;
for i=1:length(Ypred_test)
    if correct_result{i}==Ypred_test{i}
        accuracy_counter=accuracy_counter+1;
    else
        continue;
    end
end
accuracy = accuracy_counter/length(Ypred_test);
fprintf('Validation accuracy for testing group: %0.2f%%\n', accuracy*100);


%% Calculation of FAR, FRR, EER
% Load the Euclidean distance matrix
distance_matrix = distance_matrix_test_Gaussians;

% Set the threshold value for classification
threshold = 0:0.000001:0.0008;

% Initialize FAR and FRR
FAR = zeros(size(threshold));
FRR = zeros(size(threshold));

% Loop over all pairs of subjects
for t = 1:length(threshold)
    for i = 1:size(distance_matrix, 1)
        for j = 1:size(distance_matrix, 2)
            
            % If the pair of subjects belong to the same person (i.e. same class)
            % and their distance is greater than the threshold, then it is a false
            % acceptance
            if i == j && distance_matrix(i,j) > threshold(t)
                FAR(t) = FAR(t) + 1;
                
            % If the pair of subjects belong to different persons (i.e. different class)
            % and their distance is less than or equal to the threshold, then it is a false
            % rejection
            elseif i ~= j && distance_matrix(i,j) <= threshold(t)
                FRR(t) = FRR(t) + 1;
            end
            
        end
    end
    
    % Calculate FAR and FRR
    FAR(t) = FAR(t) / (size(distance_matrix, 1)^2 - size(distance_matrix, 1));
    FRR(t) = FRR(t) / (size(distance_matrix, 1)^2 - size(distance_matrix, 1));
end

% Calculate EER
diff = abs(FAR - FRR);
[~, idx] = min(diff);

EER = (FAR(idx) + FRR(idx)) / 2;

% compute the equal error rate (EER) and its threshold
EER_threshold = threshold(idx);

% plot the FAR and FRR curves
figure;
plot(threshold, FAR, 'r', 'LineWidth', 2);
hold on;
plot(threshold, FRR, 'b', 'LineWidth', 2);

% plot the EER point
plot(EER_threshold, EER, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% add grid lines, axis labels and legend
grid on;
xlabel('Threshold');
ylabel('Error Rate');
title('FAR and FRR Curves');
legend('False Acceptance Rate (FAR)', 'False Rejection Rate (FRR)', sprintf('Equal Error Rate (EER = %.2f%%)', EER*100), 'Location', 'Best');


%% functions
function unwrapped_boundary = unwrap_contour(boundary, start_idx)
    % 将轮廓重排，使得起始点在首位
    reordered_boundary = [boundary(start_idx:end, :); boundary(1:start_idx-1, :)];
    
    % 计算累积弧长
    distances = vecnorm(diff(reordered_boundary), 2, 2);
    cumulative_distances = [0; cumsum(distances)];
    
    % 将累积弧长作为 x 坐标，y 坐标保持不变
    unwrapped_boundary = [cumulative_distances, reordered_boundary(:, 2)];
end






