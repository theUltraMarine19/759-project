#include <stdio.h>
#include <iostream>
#include <fstream>


__host__ void init_membership() {
    // cout << i_cols << endl;
    // cout << i_rows << endl;
    i_membership = new float** [i_rows];
    i_new_membership = new float** [i_rows];

    for (int i = 0; i < i_rows; ++i) {
        i_membership[i] = new float* [i_cols];
        i_new_membership[i] = new float* [i_cols];
        for (int j = 0; j < i_cols; ++j) {
            i_membership[i][j] = new float[i_num_clutsers];
            i_new_membership[i][j] = new float[i_num_clutsers];
        }
        
    }

    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            for (int k = 0; k < i_num_clutsers; ++k) {
                i_membership[i][j][k] = 1 / (float)i_num_clutsers;
                i_new_membership[i][j][k] = 99999;
            }
        }
    }

    // cout << "Membership: " << endl;
    //  for (int i = 0; i < i_rows; ++i) {
    //     for (int j = 0; j < i_cols; ++j) {
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             cout << i_membership[i][j][k] << endl;;
    //         }
    //     }
    //  }

}

__host__ void init_centers() {
    i_cluster_centers = new float[i_num_clutsers];

    for (int i = 0; i < i_num_clutsers; ++i) {
        // randomly select i_num_clutsers points as cluster centers

        // random generator
        random_device rd;
        mt19937 eng(rd());
        uniform_real_distribution<> dist(0, 1);

        i_cluster_centers[i] = dist(eng);
    }

    // cout << "Centers: " << endl;
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << endl;
    // }
}

__device__ float eucl_distance(float center, float val) {
    // val: data point value
    // i: cluster center point value
    return sqrt(pow(val - center, 2));
}

__global__ void update_centers() {
    float u_ij_m, x_u_ij_m;

    for (int k = 0; k < i_num_clutsers; ++k) {
        u_ij_m = 0.0, x_u_ij_m = 0.0;
        for (int i = 0; i < i_rows; ++i) {
            for (int j = 0; j < i_cols; ++j) {
                u_ij_m += pow(i_membership[i][j][k], i_m);
                x_u_ij_m += i_image[i][j] * pow(i_membership[i][j][k], i_m);
            }
        }
        // cout << "u, x: " << u_ij_m << " " << x_u_ij_m << endl;
        // cout << "c1: " << i_cluster_centers[k] << " ";
        i_cluster_centers[k] = x_u_ij_m / u_ij_m;
        // cout << "c2: " << i_cluster_centers[k] << endl;

    }

    // cout << "Centers: ";
    // for (int i = 0; i < i_num_clutsers; ++i) {
    //     cout << i_cluster_centers[i] << " ";
    // }
    // cout <<endl;

}

__global__ float update_membership() {
    float diff = 0.0;

    // calculate degree of membership of each data point (image) regarding each cluster
    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            for (int k = 0; k < i_num_clutsers; ++k) {
                // cout << "Hi" << endl;
                // i_new_membership[i][j][k] = calculate_membership_point(i, j, k);
                i_membership[i][j][k] = calculate_membership_point(i, j, k);
            }
        }
    }

    // calculate difference between the new and old membership matrix
    diff = calculate_new_old_u_dist();

    // assign U(k + 1) to U(k)
    // for (int i = 0; i < i_rows; ++i) {
    //     for (int j = 0; j < i_cols; ++j) {
    //         for (int k = 0; k < i_num_clutsers; ++k) {
    //             // cout << "New: " << i_new_membership[i][j][k] << endl;
    //             i_membership[i][j][k] = i_new_membership[i][j][k];
    //         }    
    //     }
    // }

    // print_mebership();

    return diff;
}


__global__ float calculate_membership_point(int i, int j, int k) {
    float d_center, d_all, aggr = 0.0;

    d_center = eucl_distance(i_cluster_centers[k], i_image[i][j]);
    // cout << "d_center: " << d_center << endl;
    for (int c = 0; c < i_num_clutsers; ++c) {
        d_all = eucl_distance(i_cluster_centers[c], i_image[i][j]);
        // cout << "d_all " << c << ": " << d_all << endl;
        // cout << "center " << c << ": " << i_cluster_centers[c] << endl;
        aggr += pow((d_center / d_all), 2 / (i_m - 1));
    }

    // cout << "Aggr: " << aggr << endl;

    return 1.0 /aggr;
}



__device__ float calculate_new_old_u_dist() {
    float diff = 0.0;
    return diff;
}

__device__ void print_mebership() {
    cout << "Membership: " << endl;
    for (int i = 0; i < i_rows; ++i) {
        for (int j = 0; j < i_cols; ++j) {
            float tmp_max = -999;
            for (int k = 0; k < i_num_clutsers; ++k) {
                if (i_membership[i][j][k] >= tmp_max) {
                    // cout << "hi" << endl;
                    tmp_max = i_membership[i][j][k];
                    // i_final_cluster[i][j] = 255 / (float)(k + 1);
                    i_final_cluster[i][j] = k;
                }
                // cout << i_membership[i][j][k] << " ";
            }
            // cout << endl;
        }
    }
}

