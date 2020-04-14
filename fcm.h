#ifndef FCM_H
#define FCM_H

using namespace std;

class FCM {
    public:
        FCM(double epsilon, int data_points, int n, int num_clusters, int m);
        ~FCM();
        void init_membership();
        void init_image(double **data);
        void init_centers();
        void update_centers();
        double calculate_membership_point(int i, int j);
        double calculate_new_old_u_dist();
        double update_membership();
        double eucl_distance(int a, int b);

    private:
        double **i_image;
        double **i_membership;
        double **new_membership;
        double *i_cluster_centers;
        double i_terminate_epsilon;
        int i_num_clutsers;
        int i_data_points;
        int i_image_size;
        int i_m;

};


#endif