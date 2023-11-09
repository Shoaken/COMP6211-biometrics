%%

% input and output folder

input_folder = 'H:\biometrics\biometrics\test';

output_folder = 'H:\biometrics\biometrics\processed test';




% reading all images

file_list = dir(fullfile(input_folder, '*.jpg'));

for i = 1:88(file_list)

    % Get the filename of the current image file
    filename = fullfile(input_folder, file_list(i).name);
    I = imread(filename);

    % crop images
    I2 = imcrop(I,[500 300 1212 1371]);

    % save processed images
    [~, name, ext] = fileparts(filename);
    output_filename = fullfile(output_folder, [name '_cropped' ext]);
    imwrite(I2, output_filename);



end