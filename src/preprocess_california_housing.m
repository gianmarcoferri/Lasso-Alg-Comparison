% Preprocess California Housing Dataset for Lasso Regression

% Load raw data
raw = readtable('dataset/california_housing.csv');

% Check for missing values in total_bedrooms and fill with median
if any(ismissing(raw.total_bedrooms))
    median_bedrooms = median(raw.total_bedrooms(~ismissing(raw.total_bedrooms)));
    raw.total_bedrooms(ismissing(raw.total_bedrooms)) = median_bedrooms;
end

% Select features and target
% Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income
% Target: median_house_value

features = raw{:, 1:8};
target = raw{:, 9};

% Normalize features to [0,1]
features_norm = normalize(features, "range");

% Combine normalized features and target into a new table
processed = array2table([features_norm target], ...
    'VariableNames', [raw.Properties.VariableNames(1:8) raw.Properties.VariableNames(9)]);

% Save to new CSV
writetable(processed, 'dataset/california_housing_processed.csv');

disp('Preprocessing complete. Saved as california_housing_processed.csv in the dataset folder.');
