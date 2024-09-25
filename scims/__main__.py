#######################################################
# SCiMS: Sex calling for Metagenomic Sequences
#
# Author: Kobie Kirven 
#
# Updated by: Hanh Tran
#
# Version: 1.0.0
#
# Davenport Lab
# The Pennsylvania State University
#######################################################

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to calculate Rt values
"""
    Calculate Rt values for each scaffold in the idxstats file
"""
def calculate_Rt(idxstats, total_ref, total_map):
    rts = []
    for i in range(len(idxstats)):
        if total_map == 0 or total_ref == 0: # check for zero to avoid division by zero
            rts.append(np.nan)
        else:
            rts.append((idxstats[i][1] / total_map) / (idxstats[i][0] / total_ref))
    return np.array(rts)

def read_metadata(metadata_file):
    """Read metadata file"""
    return pd.read_csv(metadata_file, sep="\t")

def find_sample_id_column(metadata):
    """Find the sample ID column in the metadata file"""
    possible_column_names =  [
        'sample-id', 'sampleid', 'sample id',
        'id', 'featureid', 'feature id', 'feature-id', 'Run', 'SRA', 'Sample', 'sample'
    ]
    for col in possible_column_names:
        if col in metadata.columns:
            return col
    raise ValueError("No valid sample ID column found in metadata file. Expected one of: " + ", ".join(possible_column_names))

def extract_sample_id(filename, known_sample_ids):
    """Extract sample ID using the longest common substring method."""
    filename_base = os.path.splitext(filename)[0]
    if filename_base in known_sample_ids:
        return filename_base
    else:
        raise ValueError(f"Cannot find a matching sample ID for the file '{filename_base}'. Ensure filenames match sample IDs in the metadata exactly.")

def standardize_id(sample_id):
    """Standardize sample ID"""
    return sample_id.replace(" ", "_").replace(".", "_").replace("-", "_")

def match_sample_ids(metadata, results_df, sample_id_col):
    """Match sample IDs between metadata and results using standardized comparison."""
    metadata[sample_id_col] = metadata[sample_id_col].apply(standardize_id)
    results_df['Standardized SCiMS sample ID'] = results_df['SCiMS sample ID'].apply(standardize_id)
    
    # Direct match based on standardized IDs
    merged_df = pd.merge(metadata, results_df, left_on=sample_id_col, right_on='Standardized SCiMS sample ID', how='left')

    # Optionally, remove the standardized column if not needed
    merged_df.drop(columns=['Standardized SCiMS sample ID'], inplace=True)

    return merged_df

def read_master_file(master_file):
    """Read master file containing idxstats file paths"""
    with open(master_file, 'r') as file:
        idxstats_files = file.readlines()
        return [line.strip() for line in idxstats_files if line.strip()]

def calculate_posterior(prior_male, prior_female, likelihood_male_ry, likelihood_female_ry):
    """Calculate posterior probabilities for male and female sex assignment"""
    posterior_male = (prior_male * likelihood_male_ry) / (prior_male * likelihood_male_ry + prior_female * likelihood_female_ry)
    posterior_female = 1 - posterior_male
    return posterior_male, posterior_female

def determine_sex(posterior_male, posterior_female, threshold):
    """Determine sex based on posterior probabilities and threshold"""
    if posterior_male > threshold:
        return "male"
    elif posterior_female > threshold:
        return "female"
    else:
        return "uncertain"

def logit_transform(Ry):
    """Transform Ry values using logit function"""
    # to avoid issues with Ry being exactly 0 or 1, cut off values slightly below 0 or above 1
    Ry = np.clip(Ry, 1e-9, 1 - 1e-9)
    return np.log(Ry / (1 - Ry))

def probit_transform(Ry):
    """Transform Ry values using probit function to apply the inverse CDF of a normal distribution"""
    Ry = np.clip(Ry, 1e-9, 1 - 1e-9)
    return norm.ppf(Ry)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sex Assignment Script')
    parser.add_argument('--scaffolds', dest="scaffold_ids_file", required=True, type=str, help='File containing scaffold IDs of interest')
    parser.add_argument('--metadata', required=True, help="Path to the metadata file")
    parser.add_argument('--master_file', required=True, help="Path to the master file with idxstats paths")
    parser.add_argument('--homogametic_id', dest="x_id", required=True, type=str, help='Specify scaffold ID for homogametic chromosome (eg. In XY sex determination systems, homogametic chromosome is X,while in ZW sex determination systems, homogametic chromosome is Z )')
    parser.add_argument('--heterogametic_id', dest="y_id", required=True, type=str, help='Specify scaffold ID for heterogametic chromosome (eg. In XY sex determination systems, heterogametic chromosome is Y,while in ZW sex determination systems, heterogametic chromosome is W)')
    parser.add_argument('--system', dest="system", required=True, type=str, choices=['XY', 'ZW'], help='Specify the host sex determination system (XY or ZW)')
    parser.add_argument('--threshold', dest="threshold", type=float, default=0.95, help='Probability threshold for determining sex (default: 0.95)')
    parser.add_argument('--output', dest="output_file", required=True, type=str, help='Output file to save the results')
    parser.add_argument('--training_data', dest="training_data", type=str, default="training_data/training_data_ry.txt", help='Path to the training data file if you have your own training data for your organism of interest')
    args = parser.parse_args()

    print("=================================================")
    print("""
                                                    
    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|  
    _|        _|          _|    _|_|  _|_|  _|        
    _|_|    _|          _|    _|  _|  _|    _|_|    
        _|  _|          _|    _|      _|        _|  
    _|_|_|      _|_|_|  _|_|_|  _|      _|  _|_|_|    
    """)
    print("=================================================")
    # Load the training data
    training_data = pd.read_csv(args.training_data, sep='\t')
    logging.info(f"Training data is loaded successfully")

    # Flip the sex labels if the system is ZW
    if args.system == 'ZW':
        training_data['actual_sex'] = training_data['actual_sex'].apply(lambda x: 'female' if x == 'male' else 'male')
        logging.info(f"Training data is adjusted for ZW system")

    # Separate Ry values by known sex
    ry_values_male = training_data[training_data['actual_sex'] == 'male']['Ry'].values
    ry_values_female = training_data[training_data['actual_sex'] == 'female']['Ry'].values

    # Transform Ry before applying Gaussian KDE
    ry_values_male_transformed = logit_transform(ry_values_male)
    ry_values_female_transformed = logit_transform(ry_values_female)

    kde_ry_male_transformed = gaussian_kde(ry_values_male_transformed)
    kde_ry_female_transformed = gaussian_kde(ry_values_female_transformed)

    # Load metadata
    metadata = read_metadata(args.metadata)
    logging.info(f"Metadata is loaded successfully")
    sample_id_col = find_sample_id_column(metadata)
    known_sample_ids = metadata[sample_id_col].tolist()  # List of known sample IDs
    idxstats_files = read_master_file(args.master_file)

    results = []

    with open(args.scaffold_ids_file, 'r') as file:
        scaffold_ids = file.read().splitlines()

    for idxstats_file in idxstats_files:
        sample_id = extract_sample_id(os.path.basename(idxstats_file), known_sample_ids)
        sample_id = standardize_id(sample_id)  # Standardize extracted sample ID

        idxstats = pd.read_table(idxstats_file, header=None, index_col=0)
        idxstats_filtered = idxstats.loc[scaffold_ids]

        sex_chromosomes = [args.x_id, args.y_id]
        autosomes = set(scaffold_ids) - set(sex_chromosomes)

        total_ref = idxstats_filtered.iloc[:, 0].sum()
        total_map = idxstats_filtered.iloc[:, 1].sum()

        # Calculate Rt values
        Rt_values = calculate_Rt(idxstats_filtered.values[:, :2], total_ref, total_map)

        if args.system == 'XY':
            x_id = args.x_id
            y_id = args.y_id

            # Ensure that x_id is in the filtered idxstats
            if x_id not in idxstats_filtered.index:
                raise ValueError(f"{x_id} not found in the filtered idxstats")

            # Get the index of the X chromosome
            x_index = idxstats_filtered.index.get_loc(x_id)

            # Get the index of the Y chromosome if it exists
            y_index = idxstats_filtered.index.get_loc(y_id) if y_id in idxstats_filtered.index else None

            # Calculate Rx by comparing the X chromosome to the mean of the autosomes
            autosomal_Rt_values = np.array([Rt_values[i] for i in range(len(Rt_values)) if i != x_index and (y_index is None or i != y_index)]) # get rt value for each autosome
            tot = Rt_values[x_index]/autosomal_Rt_values # calculate ratio of each X:autosome pair
            Rx = np.mean(tot) # calculate mean of all X:autosome pairs

            # Calculate Rx and CIs
            z_value = np.round(norm.ppf(1 - args.threshold), 3)
            conf_interval = (np.std(autosomal_Rt_values) / np.sqrt(len(autosomal_Rt_values))) * z_value
            CI1_Rx, CI2_Rx = sorted([Rx - conf_interval, Rx + conf_interval])

            # Calculate Ry and CIs
            x_count = idxstats.loc[x_id].iloc[1] # Number of reads mapped to X chromosome
            y_count = idxstats.loc[y_id].iloc[1] # Number of reads mapped to Y chromosome
            tot_xy = x_count + y_count # Total number of reads mapped to Y chromosome    

            if tot_xy == 0:
                # Handle division by zero
                Ry = np.nan
                conf_interval = np.nan
                CI1_y = np.nan
                CI2_y = np.nan
                inferred_sex = 'uncertain'
                posterior_male = np.nan
                posterior_female = np.nan
            else:
                Ry = (1.0 * y_count) / tot_xy
                conf_interval = z_value * (np.sqrt((Ry * (1 - Ry)) / tot_xy))
                CI1_y, CI2_y = sorted([Ry - conf_interval, Ry + conf_interval])

                # When applying KDE on new samples, transform the new Ry values using the same logit transform
                Ry_transformed = logit_transform(Ry)

                # Validate Ry_transformed
                if not np.isfinite(Ry_transformed):
                    # Handle invalid transformed Ry
                    inferred_sex = 'uncertain'
                    posterior_male = np.nan
                    posterior_female = np.nan
                else:
                    # Apply KDE to the transformed values
                    likelihood_male_Ry = kde_ry_male_transformed.evaluate([Ry_transformed])[0]
                    likelihood_female_Ry = kde_ry_female_transformed.evaluate([Ry_transformed])[0]

                    prior_male = 0.5
                    prior_female = 0.5

                    # Use KDE models for posterior calculation
                    posterior_male, posterior_female = calculate_posterior(prior_male, prior_female, likelihood_male_Ry, likelihood_female_Ry)
                    
                    # Determine sex
                    inferred_sex = determine_sex(posterior_male, posterior_female, threshold=args.threshold)

            result = {
                'SCiMS sample ID': sample_id,
                'Total reads mapped': total_map,
                'Total reads mapped X/Z': x_count,
                'Total reads mapped Y/W': y_count,
                'SCiMS predicted sex': inferred_sex,
                'Posterior probability of being male': float(np.round(posterior_male, 4)),
                'Posterior probability of being female': float(np.round(posterior_female, 4)),
                'Status': 'Success'
            }

            if args.system == 'XY':
                result['Rx'] = float(np.round(Rx, 4)) if np.isfinite(Rx) else np.nan
                result['Rx 95% CI'] = (float(np.round(CI1_Rx, 3)), float(np.round(CI2_Rx, 3))) if np.isfinite(CI1_Rx) and np.isfinite(CI2_Rx) else (np.nan, np.nan)
                result['Ry'] = float(np.round(Ry, 4)) if np.isfinite(Ry) else np.nan
                result['Ry 95% CI'] = (float(np.round(CI1_y, 3)), float(np.round(CI2_y, 3))) if np.isfinite(CI1_y) and np.isfinite(CI2_y) else (np.nan, np.nan)

        else:  # ZW system
            z_id = args.x_id
            w_id = args.y_id

            if z_id not in idxstats.index:
                raise ValueError(f"{z_id} not found in the filtered idxstats")

            z_index = idxstats.index.get_loc(z_id)
            w_index = idxstats.index.get_loc(w_id) if w_id in idxstats.index else None

            # Calculate Rx by comparing the X chromosome to the mean of the autosomes
            autosomal_Rt_values = np.array([Rt_values[i] for i in range(len(Rt_values)) if i != z_index and (w_index is None or i != w_index)])
            tot = Rt_values[z_index]/autosomal_Rt_values # calculate ratio of each Z:autosome pair
            
            # Calculate Rz and CIs
            Rz = np.mean(tot) # calculate mean of all Z:autosome pairs

            z_value = np.round(norm.ppf(1 - args.threshold), 3)
            conf_interval = (np.std(autosomal_Rt_values) / np.sqrt(len(autosomal_Rt_values))) * z_value
            CI1_Rz, CI2_Rz = sorted([Rz - conf_interval, Rz + conf_interval])

            # Calculate Rw and CIs
            z_count = idxstats.loc[z_id].iloc[1] # Number of reads mapped to Z chromosome
            w_count = idxstats.loc[w_id].iloc[1] # Number of reads mapped to W chromosome
            tot_zw = z_count + w_count # Total number of reads mapped to W chromoso e

            if tot_zw == 0:
                # Handle division by zero
                Rw = np.nan
                conf_interval = np.nan
                CI1_w = np.nan
                CI2_w = np.nan
                inferred_sex = 'uncertain'
                posterior_male = np.nan
                posterior_female = np.nan
            else:
                Rw = (1.0 * w_count) / tot_zw

                conf_interval = z_value * (np.sqrt((Rw * (1 - Rw)) / tot_zw))
                CI1_w, CI2_w = sorted([Rw - conf_interval, Rw + conf_interval])

                # When applying KDE on new samples, transform the new Ry values using the same logit transform
                Ry_transformed = logit_transform(Rw)  # Note: Using Rw for ZW system
        
                # Validate Ry_transformed
                if not np.isfinite(Ry_transformed):
                    # Handle invalid transformed Ry
                    inferred_sex = 'uncertain'
                    posterior_male = np.nan
                    posterior_female = np.nan
                else:
                    # Apply KDE to the transformed values
                    likelihood_male_Ry = kde_ry_male_transformed.evaluate([Ry_transformed])[0]
                    likelihood_female_Ry = kde_ry_female_transformed.evaluate([Ry_transformed])[0]

                    prior_male = 0.5
                    prior_female = 0.5
                
                    # Use KDE models for posterior calculation
                    posterior_male, posterior_female = calculate_posterior(prior_male, prior_female, likelihood_male_Ry, likelihood_female_Ry)
                
                    inferred_sex = determine_sex(posterior_female, posterior_male, threshold=args.threshold)

            result = {
                'SCiMS sample ID': sample_id,
                'Total reads mapped': total_map,
                'Total reads mapped X/Z': z_count,
                'Total reads mapped Y/W': w_count,
                'SCiMS predicted sex': inferred_sex,
                'Posterior probability of being male': float(np.round(posterior_male, 4)),
                'Posterior probability of being female': float(np.round(posterior_female, 4)),
                'Status': 'Success'
            }

            result['Rz'] = float(np.round(Rz, 4)) if np.isfinite(Rz) else np.nan
            result['Rz 95% CI'] = (float(np.round(CI1_Rz, 3)), float(np.round(CI2_Rz, 3))) if np.isfinite(CI1_Rz) and np.isfinite(CI2_Rz) else (np.nan, np.nan)
            result['Rw'] = float(np.round(Rw, 4)) if np.isfinite(Rw) else np.nan
            result['Rw 95% CI'] = (float(np.round(CI1_w, 3)), float(np.round(CI2_w, 3))) if np.isfinite(CI1_w) and np.isfinite(CI2_w) else (np.nan, np.nan)
        
        results.append(result)

    results_df = pd.DataFrame(results)
    # Merging with metadata
    merged_df = match_sample_ids(metadata, results_df, sample_id_col)

    # Save the results
    merged_df.to_csv(args.output_file, sep='\t', index=False)
    logging.info(f"Results are saved in {args.output_file}")

if __name__ == "__main__":
    main()
