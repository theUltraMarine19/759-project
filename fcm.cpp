#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include "fcm.h"

FCM::FCM(double epsilon, int data_points, int n, int num_clusters, int m) {
    i_terminate_epsilon = epsilon;
    i_membership = nullptr;
    i_cluster_centers = nullptr;
    i_image = nullptr;
    i_num_clutsers = num_clusters;
    i_data_points = data_points;
    i_m = m;
    i_image_size = n;
}

FCM::~FCM() {
    if (i_image != nullptr) {
        delete i_image;
    }

    if (i_cluster_centers != nullptr) {
        delete i_cluster_centers;
    }

    if (i_membership != nullptr) {
        delete i_membership;
    }
}

void FCM::init_image(double **data) {
    i_image = new double* [i_image_size];
    for (int i = 0; i < i_image_size; ++i) {
        i_image[i] = new double[i_image_size];
    }

    for (int i = 0; i < i_image_size; ++i) {
        for (int j = 0; j < i_image_size; ++j) {
            i_image[i][j] = data[i][j];
        }
    }
}

void FCM::init_membership() {
    i_membership = new double* [i_data_points];

    for (int i = 0; i < i_data_points; ++i) {
        i_membership[i] = new double [i_num_clutsers];
    }

    for (int i = 0; i < i_data_points; ++i) {
        memset(i_membership[i], 1 / i_num_clutsers, sizeof(i_membership[i]));
    }
}

void FCM::init_centers() {
    for (int i = 0; i < i_num_clutsers; ++i) {
        // random select i_num_clutsers points as cluster centers
    }
}

double FCM::eucl_distance(int a, int b) {
    // calculate distance to the centers

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

double FCM::calculate_new_old_u_dist() {
    double diff = 0.0;
    return diff;
}

void FCM::update_centers() {
    for (int j = 0; j < i_num_clutsers; ++j) {
        double u_ij_m = 0.0, x_u_ij_m = 0.0;
        // calculate sum of u^m and the sum of x * u^m
        for (int i = 0; i < i_data_points; ++i) {
            u_ij_m += pow(i_membership[i][j], i_m);
            x_u_ij_m += i_image[i][j] * pow(i_membership[i][j], i_m);
        }
        i_cluster_centers[j] = x_u_ij_m / u_ij_m;
    }
}

