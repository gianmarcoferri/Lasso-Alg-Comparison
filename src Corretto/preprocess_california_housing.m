% Preprocessing del Dataset California Housing Prices 

% Caricamento del file CSV originale e conversione in una tabella MATLAB.
raw = readtable('dataset/california_housing.csv');

% Controllo se nella colonna total_bedrooms ci sono valori mancanti.
if any(ismissing(raw.total_bedrooms))
    
    % Calcolo della mediana dei valori non mancanti
    median_bedrooms = median(raw.total_bedrooms(~ismissing(raw.total_bedrooms)));
    
    % Sostituzione dei valori mancanti con la mediana
    raw.total_bedrooms(ismissing(raw.total_bedrooms)) = median_bedrooms;
end

% Selezione delle prime 8 colonne come features:
% longitude, latitude, housing_median_age, total_rooms,
% total_bedrooms, population, households, median_income
% L'ultima (colonna 9) è il target: median_house_value

features = raw{:, 1:8}; % Estrazione dei valori numerici delle feature
target = raw{:, 9}; % Estrazione della colonna target

% Normalizzazione min-max delle feature in [0,1]
features_norm = normalize(features, "range");

% Combinazione delle feature normalizzate e del target in una nuova tabella
processed = array2table([features_norm target], ...
    'VariableNames', [raw.Properties.VariableNames(1:8) raw.Properties.VariableNames(9)]);

% Salvataggio del dataset preprocessato in un nuovo file CSV
writetable(processed, 'california_housing_processed.csv');

% Messaggio finale
disp('Preprocessing complete. Saved as california_housing_processed.csv');
