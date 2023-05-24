% Computer Vision Project
% Authors - Adish Pathare, Aditya Tirakannavar, Disha Revandkar.


function project_cv
    % random seed to keep consistent results.
    rng(1);
    % load the image files from image folder.
    [image_files, num_images] = load_images();
    % Split the given images into training and testing set
    % We have followed a 80-20 split (80 training and 20 Test)
    [training_images, testing_images, training_indices, testing_indices] =...
        train_test_split(image_files, num_images);
    % Load the labels 
    labels = load_labels();
    % Extract training features and training labels.
    [training_features, training_labels] = training_features_extraction...
    (training_images, labels, training_indices);
    % Train the classification model
    classification_model = train_model(training_features, training_labels);
    % Predict the labels of test images.
    [num_correct, correct_labels, predicted_labels] = ...
        testing_process(testing_images, testing_indices, labels, classification_model);
    % Show the final results
    show_results(correct_labels, testing_images)
    % Calculate the accuracy
    accuracy = calculate_accuracy(num_correct, testing_images);
    disp(['Accuracy: ' num2str(accuracy * 100) '%']);
    % Display the confusion matrix
    confusion_matrix(labels, testing_indices, predicted_labels);
end

function show_results(correct_labels, testing_images)
    % Show results of random classified images in a subplot.
    permuted_indices = randperm(length(correct_labels));
    selected_indices = permuted_indices(1:25);
    figure
    for i = 1:25
        image_index = correct_labels(selected_indices(i),1);
        label_cotton = correct_labels(selected_indices(i),2);
        if label_cotton == 1
            label = 'Rainfed';
        elseif label_cotton == 2
            label = 'Fully irrigated';
        elseif label_cotton == 3
            label = 'Percent deficit';
        elseif label_cotton == 4
            label = 'Time delay';
        end
        subplot(5,5,i); 
        imshow(testing_images{image_index});
        title(label, 'FontSize', 8);
    end
end

function accuracy = calculate_accuracy(num_correct, testing_images)
    % Calculate the accuracy.
    accuracy = num_correct / length(testing_images);
end

function [image_files, num_images] = load_images()
    % Load image files.
    image_files = dir(fullfile("cotton images and labels/cotton images/", '*.TIF'));
    num_images = length(image_files);
end

function labels = load_labels()
    % Load labels.
    labels = csvread('labels_rgb.csv', 1, 0);
end

function [training_images, testing_images, training_indices, testing_indices] = ...
    train_test_split(image_files, num_images)
    % Use 80% of images for training
    training_indices = randperm(num_images, round(0.8 * num_images));
    testing_indices = setdiff(1:num_images, training_indices);
    testing_images = cell(length(testing_indices), 1);
    training_indices = [training_indices randsample(testing_indices, ...
        round(0.5 * numel(testing_indices)))];
    training_images = cell(length(training_indices), 1);

    % Get training images and convert to double for efficient processing.
    for i = 1:length(training_indices)
        filename = fullfile("cotton images and labels/cotton images/",...
            image_files(training_indices(i)).name);
        training_images{i} = imread(filename);
        training_images{i} = training_images{i}(:,:,1:3);
        training_images{i} = im2double(training_images{i});
    end
    
    % Get testing images and convert to double for efficient processing.
    for i = 1:length(testing_indices)
        filename = fullfile("cotton images and labels/cotton images/",...
            image_files(testing_indices(i)).name);
        testing_images{i} = imread(filename);
        testing_images{i} = testing_images{i}(:,:,1:3);
        testing_images{i} = im2double(testing_images{i});
    end
end

function [training_features, training_labels] = training_features_extraction...
    (training_images, labels, training_indices)
    % Extract features from training images
    training_features = zeros(length(training_images), 3);
    training_labels = zeros(length(training_images),1);
    for i = 1:length(training_images)
        im_in = training_images{i};
        % Extract the number of cottons, the greenary of the grass and the
        % area of the grass.
        [num_cotton, plant_green_texture, cotton_plant_area] =...
            extract_image_features(im_in);
        % Associate the features with each training image.
        training_features(i, :) = ...
            [num_cotton, plant_green_texture, cotton_plant_area];
        % Get the corresponding labels for training the model.
        training_labels(i) = labels(training_indices(i),2);
    end
end

function [num_cotton, plant_green_texture, cotton_plant_area] = ...
        extract_image_features(im_in)

    % Use color segmentation to isolate the green parts of the image
    hsv_img = rgb2hsv(im_in);
    green_mask = hsv_img(:,:,1) > 0.2 & hsv_img(:,:,1) < 0.5 &...
        hsv_img(:,:,2) > 0.3 & hsv_img(:,:,3) > 0.05;
    brown_mask = ~green_mask;
 
    % Remove small objects from the binary masks
    se = strel('disk', 10);
    green_mask = imopen(green_mask, se);
    brown_mask = imopen(brown_mask, se);
    
    % Use edge detection to find the edges of the green and brown masks
    green_edges = edge(green_mask, 'Canny', [0.05, 0.2]);
    brown_edges = edge(brown_mask, 'Canny', [0.05, 0.2]);
    
    % Combine the green and brown edges to obtain the 
    % edges of the cotton plant
    plant_edges = green_edges | brown_edges;
 
    % Segment plant from the brownish background.
    height = size(plant_edges, 1);
    line1_y = round(0.02 * height);
    line2_y = round(0.98 * height);
    plant_edges(line1_y, :) = true;
    plant_edges(line2_y, :) = true;
    width = size(plant_edges, 2);
    line_x = round(0.02 * width);
    line_y1 = round(0.05 * size(plant_edges, 1));
    line_y2 = round(0.95 * size(plant_edges, 1));
    plant_edges(line_y1:line_y2, line_x) = true;
    plant_edges(line_y1:height, width-line_x+1) = true;
    
    % Fill the plant edges to get the plant shape.
    plant_structure = imfill(plant_edges, 'holes');
    
    % Use the above mask to extract the plant from original image.
    plant_structure(line1_y, :) = false;
    plant_structure(line2_y, :) = false;
    plant_structure(line_y1:line_y2, line_x) = false;
    plant_structure(line_y1:height, width-line_x+1) = false;
    
    % Original image with the brown background deleted.
    original_with_mask = im_in .* cat(3, plant_structure,...
        plant_structure, plant_structure);
    
    % Convert to cotton plant to grayscale for further processing.
    im_gray = rgb2gray(original_with_mask);
    binaryImage = im_gray > 0.75;
    % Get the cotton buds from the plant.
    se = strel('disk', 5);
    eliminate_noise = imclose(binaryImage, se);
    % Fill the holes to get the cotton shape
    cottons_segmented = imfill(eliminate_noise, 'holes');
    % Detect and count the number of cottons in the image
    stats = regionprops(cottons_segmented, 'Area');
    % Assume cottons are disk shaped and area more than 8.
    num_cotton = length(find([stats.Area] > 8));
    % Get the area of the plant.
    stats = regionprops(plant_structure, 'Area');
    % Max area since the plant will be the largest connected component.
    cotton_plant_area = max([stats.Area]);
    green_channel = original_with_mask(:,:,2);
    total_greenary = sum(green_channel(:));
    % Calculate the average of the pixel values in the green channel to
    % get the green texture of the image.
    plant_green_texture = total_greenary / numel(green_channel);
end

function classification_model = train_model(training_features,...
    training_labels)
    % Use a random forest classifier with 50 decision trees.
    num_trees = 50;
    % Train the classifier.
    classification_model = TreeBagger(num_trees,...
        training_features, training_labels(:, 1));
end

function [num_correct, correct_labels, predicted_labels] = ...
        testing_process(testing_images, testing_indices, labels,...
        classification_model)
    % Keeps count of correct predictions.
    num_correct = 0;
    % Keeps the correct labels.
    correct_labels = [];
    % Keeps the predicted labels.
    predicted_labels = [];
    for i = 1:length(testing_images)
        im_in = testing_images{i};
        % Extract features of test image similar to the training image.
        [num_cotton, plant_green_texture, cotton_plant_area] =...
            extract_image_features(im_in);
        test_features = [num_cotton, plant_green_texture, cotton_plant_area];
        % Classify cotton image using the trained classifier into the
        % correct irrigation type.
        label_cotton = predict(classification_model, test_features);
        % The random forest classifier returns a cell array.
        label_cotton = str2double(label_cotton{1});
        predicted_labels(i) = label_cotton;
        if label_cotton == labels(testing_indices(i), 2)
            % If the prediction is correct execute the below commands.
            num_correct = num_correct + 1;
            correct = [i, label_cotton];
            correct_labels = [correct_labels; correct];
        end
    end
end

function confusion_matrix(labels, testing_indices, predicted_labels)
    % Display the confusion matrix.
    figure
    confusion = confusionmat(labels(testing_indices, 2), predicted_labels);
    confusionchart(confusion);
end