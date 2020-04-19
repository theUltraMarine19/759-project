#ifndef FCM_H
#define FCM_H

using namespace std;

class FCM {
    public:
        FCM(float** img, float epsilon, int rows, int cols, int num_clusters, int m);
        ~FCM();
        void init_membership();
        void init_centers();
        void update_centers();
        float calculate_membership_point(int i, int j, int k);
        float calculate_new_old_u_dist();
        float update_membership();
        float eucl_distance(int i, int k);

    private:
        float **i_image;
        float ***i_membership;
        float ***i_new_membership;
        float *i_cluster_centers;
        double i_terminate_epsilon;
        int i_rows, i_cols;
        int i_num_clutsers;
        int i_image_size;
        int i_m;
        bool done;

};


#endif