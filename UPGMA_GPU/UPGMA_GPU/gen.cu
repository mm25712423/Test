#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    srand(time(NULL));
    int N;
    int i,j;
    FILE *fp;
    printf("Input gen size : ");
    scanf("%d",&N);
    char filename[10];
    sprintf(filename,"%d.txt", N); 
    fp = fopen(filename,"w");
    fprintf(fp,"%d\n",N);
    int **UPCluster;
    UPCluster = (int **)malloc((N+1)*sizeof(int*));
    for(i=0;i<(N+1);i++)
    {
            UPCluster[i] =(int*)malloc( N * sizeof(int));
    }
    //cudaMallocManaged(&UPCluster, (N+1) * sizeof(int));
    //for(i=0;i<(N+1);i++)
    //{
    //     cudaMallocManaged(&UPCluster[i], N*sizeof(int));
    //}

    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            UPCluster[i][j] = (rand()%1000)+1;
            if(i == j)  UPCluster[i][j] = 0;
            else if(i > j) UPCluster[i][j] = UPCluster[j][i];
            //printf("%4d ",UPCluster[i][j]);
        }
        //printf("\n");
    }

    for(i = 0; i < N; i++)
    {
        fprintf(fp,"%d ",i);
        for(j = 0; j < N; j++){

                fprintf(fp,"%d ",UPCluster[i][j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

    free(UPCluster);
}
