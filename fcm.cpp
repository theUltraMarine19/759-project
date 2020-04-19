#include <iostream>
#include <float.h>
#include <math.h>
#include <random>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fcm.h"

using namespace std;

FCM::FCM(float** img, float epsilon, int rows, int cols, int num_clusters, int m) {
    i_terminate_epsilon = epsilon;
    i_membership = nullptr;
    i_cluster_centers = nullptr;
    i_image = img;
    i_num_clutsers = num_clusters;
    i_rows = rows;
    i_cols = cols;
    i_m = m;
}

FCM::~FCM() {
    if (i_image != nullptr) {
        delete[] i_image;
    }

    if (i_cluster_centers != nullptr) {
        delete i_cluster_centers;
    }

    if (i_membership != nullptr) {
        delete[] i_membership;
    }

    if (new_membership != nullptr) {
        delete[] new_membership;
    }
}


void FCM::init_membership() {
    i_membership = new float* [i_rows];

    for (int i = 0; i < i_rows; ++i) {
        i_membership[i] = new float[i_cols];
    }

    for (int i = 0; i < i_rows; ++i) {
        memset(i_membership[i], 1 / i_num_clutsers, sizeof(i_membership[i]));
    }

}

void FCM::init_centers() {
    i_cluster_centers = new float[i_num_clutsers];

    for (int i = 0; i < i_num_clutsers; ++i) {
        // random select i_num_clutsers points as cluster centers

        // random generator
        random_device rd;
        mt19937 eng(rd());
        uniform_real_distribution<> dist(0, 1);

        i_cluster_centers[i] = dist(eng);
    }
}

double FCM::eucl_distance(int i, int val) {
    // val: data point value
    // i: cluster center point
    return sqrt(pow(val - i_cluster_centers[i], 2));
}

void FCM::update_centers() {
    double u_ij_m, x_u_ij_m;

    for (int k = 0; k < i_num_clutsers; ++k) {
        u_ij_m = 0.0, x_u_ij_m = 0.0;
        for (int i = 0; i < i_rows; ++i) {
            for (int j = 0; j < i_cols; ++j) {
                u_ij_m += pow(i_membership[i][j], i_m);
                x_u_ij_m += i_image[i][j] * pow(i_membership[i][j], i_m);
            }
        }
        i_cluster_centers[k] = x_u_ij_m / u_ij_m;
    }
}

double FCM::update_membership() {
    double diff = 0.0;

    // check if this is the first iteration
    if (i_membership == nullptr) {
        init_membership();
    }

    // calculate degree of membership of each data point (image) regarding each cluster
    for (int i = 0; i < i_data_points; ++i) {
        for (int j = 0; j < i_num_clutsers; ++j) {
            new_membership[i][j] =  calculate_membership_point(i, j);
        }
    }

    // calculate difference between the new and old membership matrix
    diff = calculate_new_old_u_dist();

    // assign U(k + 1) to U(k)
    // this could be a huge hit in performance...
    for (int i = 0; i < i_data_points; ++i) {
        for (int j = 0; j < i_num_clutsers; ++j) {
            i_membership[i][j] = new_membership[i][j];
        }
    }

    return diff;
}


double FCM:: calculate_membership_point(int i, int j) {
    // i: data #
    // j: cluster #
    double d_ij, d_ik, exp, aggr = 0.0;
    for (int k = 0; k < i_num_clutsers; ++k) {
        d_ik = eucl_distance(i, k);
        d_ij = eucl_distance(i, j);
        exp = pow((d_ij / d_ik), 2 / (i_m - 1));
        aggr += exp;
    }

    return 1 /aggr;

}



double FCM::calculate_new_old_u_dist() {
    double diff = 0.0;
    return diff;
}

