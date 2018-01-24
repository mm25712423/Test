#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

const char *sSDKsample = "threadFenceReduction";
#if CUDART_VERSION >= 2020
#include "threadFenceReduction_kernel.cuh"
#else
#pragma comment(user, "CUDA 2.2 is required to build for threadFenceReduction")
#endif

unsigned int nextPow2(unsigned int x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
}
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
        if (n == 1)
        {
                threads = 1;
                blocks = 1;
        }
        else
        {
                threads = (n < maxThreads * 2) ? nextPow2(n / 2) : maxThreads;
                blocks = max(1, n / (threads * 2));
        }

        blocks = min(maxBlocks, blocks);
}

typedef struct node *TREE_ptr;
struct node
{
        int NODE;
        int LEVEL;
        TREE_ptr ancestor, left_child, right_child;
        int left_level, right_level;
};
int Max(int x,int y)
{
   return  (x>y)? x:y;
}

void trans(float **UPCluster,int N,float *UP1D,int i_min,int j_min)
{
        int i;
        float x;

        for(i=0;i<N;i++)
        {
                int IN = i*N;
                x = UP1D[i_min*N+i] = UPCluster[i_min][i];
                UP1D[IN+i_min] = x;
                UP1D[IN+j_min] = 0;
                UP1D[j_min*N+i] = 0;
        }

}
void preorder(TREE_ptr ptr,int N)
{
    //printf("1");
    if(ptr)
    {
               {
                    if(ptr->ancestor!=NULL)
                    {
                        if(ptr->NODE==ptr->ancestor->right_child->NODE)
                        ptr->right_level=ptr->ancestor->right_level+1;
                        else
                        ptr->left_level=ptr->ancestor->left_level+1;
                    }
               }
               preorder(ptr->left_child,N);
               preorder(ptr->right_child,N);
    }
}

void Inoder_Result(TREE_ptr ptr,int N)
{
        int i;
        if(ptr)
        {
                Inoder_Result(ptr->left_child,N);

                if(ptr->ancestor!=NULL)
                {
                //fprintf(ftree,"%d ",ptr->NODE);
                //fprintf(ftree,"%d ",ptr->LEVEL);
                //fprintf(ftree,"%d ",ptr->ancestor->LEVEL);
                }

                if(ptr->NODE >= N)
                {
                        printf(",");
                        //fprintf(fw,",");
                }
                else
                {
                        for(i=0;i<ptr->left_level;i++)
                        {
                        printf("(");
                        //fprintf(fw,"(");
                        }
                        printf("%d",ptr->NODE);
                        //fprintf(fw,"%d",ptr->NODE);

                        for(i=0;i<ptr->right_level;i++)
                        {
                        printf(")");
                        //fprintf(fw,")");
                        }
                }
                Inoder_Result(ptr->right_child,N);
        }
}


void node_Initial(struct node *node, int N)
{
    int i;
    for(i=0;i<2*N-1;i++)
    {
        node[i].NODE=i;
        node[i].LEVEL=0;
        node[i].ancestor=NULL;
        node[i].left_child=NULL;
        node[i].right_child=NULL;
        node[i].left_level=0;
        node[i].right_level=0;
    }
}

void Build_Tree(struct node *node,int i,int j,int K)
{
        //when node[i].ancestor is NULL and node[j].ancestor is NULL
        if((!node[i].ancestor)&&(!node[j].ancestor))
        {
            node[K].left_child  = &(node[i]);
            node[K].right_child = &(node[j]);
        }
        //when node[i].ancestor isn't NULL and node[j].ancestor is NULL
        else if((node[i].ancestor)&&(!node[j].ancestor))
        {
            node[K].left_child  = node[i].ancestor;
            node[K].right_child = &(node[j]);
        }
        //when node[i].ancestor is NULL and node[j].ancestor isn't NULL
        else if(!(node[i].ancestor)&&(node[j].ancestor))
        {
            node[K].left_child  = &(node[i]);
            node[K].right_child = node[j].ancestor;
        }
        //when both node[i].ancestor and node[j].ancestor are not NULL
        else
        {
            node[K].left_child  = node[i].ancestor;
            node[K].right_child = node[j].ancestor;
        }
        node[i].ancestor = node[j].ancestor = &(node[K]);
        node[K].LEVEL = Max(node[i].LEVEL,node[j].LEVEL)+1;

}
void show(float **UPCluster,int N){
    int i,j;
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            printf("%3.0f ",UPCluster[i][j]);
        }
        printf("\n");
    }
}
void to2D(float *UP1D, float **UPCluster, int N){
    int i,j;
    
    for(i=0;i<N;i++)
        {
                for(j=0;j<N;j++)
                {
                        UPCluster[i][j] = UP1D[i*N+j];
                }
        }    
}

__global__ void update(float *UP1D, int *idx_result,float *tmp, int N, int K, int *c_node)
{
    //extern __shared__ float tmp[];

    int i_min = 0;
    int j_min = 0;
    int inver = 0;

    i_min = idx_result[0] % N;
    j_min = idx_result[0] / N;
    inver = i_min * N + j_min;

    if(inver < idx_result[0]){

        idx_result[0] = inver;
        i_min = idx_result[0] % N;
        j_min = idx_result[0] / N;

    }
    c_node[2*(K-N)] = i_min;
    c_node[2*(K-N)+1] = j_min;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //update
    //printf("%d\n",tid);

    while(tid < N){
        
        //printf("1tid=%d N=%d\n",tid,N);    
        //UPCluster[N][t] = (UPCluster[i_min][t] + UPCluster[j_min][t])/2.0;
        int TN = tid*N;
        int JNT = j_min*N+tid;
        int INT = i_min*N+tid;

        //cal temp for replace
        tmp[tid] = (UP1D[INT] + UP1D[JNT])/2;
        tmp[i_min] = 0;
        tmp[j_min] = 0;
        //__syncthreads();
        //printf("2tid=%d N=%d\n",tid,N);
        //del
        //UP1D[INT] = 0;
        UP1D[JNT] = 0;
        UP1D[TN+j_min] = 0;
        //UP1D[TN+i_min] = 0;
        //__syncthreads();
        //printf("3tid=%d N=%d\n",tid,N);
        //replcae
        UP1D[INT] = tmp[tid];//OK
        UP1D[TN+i_min] = tmp[tid];
        //__syncthreads();
        //printf("4tid=%d N=%d\n",tid,N);

        tid += gridDim.x * blockDim.x;
        //printf("%d\n",tid);
    }
    //printf("%d-\n,",tid);
    __syncthreads();
}
void cpu(int N,int K,float *UP1D,int *ind,struct node *node, int *cid, float *h_UP1D)
{
        int i_min, j_min;
        int size = N * N; 

        int maxThreads = 512;
        int maxBlocks = 32;
        int numBlocks = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);
        printf("threads %d blocks %d\n",numThreads,numBlocks);
        //save node
        int *h_c_node;
        int *c_node;
        h_c_node = (int *)malloc(2 * N * sizeof(int));
        cudaMalloc((void**)&c_node, 2 * N * sizeof(int));      
        //cudaMallocManaged(&c_node, 2 * N * sizeof(int));  

        int *idx_result;
        cudaMallocManaged(&idx_result, numBlocks * sizeof(int));
        //cudaMalloc((void**)&idx_result, numBlocks * sizeof(int));   

        float *dev_d_odata;
        //cudaMallocManaged(&dev_d_odata, numBlocks * sizeof(float));
        cudaMalloc((void**)&dev_d_odata, numBlocks * sizeof(int));   

        float *tmp;
        cudaMalloc((void**)&tmp, N * sizeof(float));


        cudaEvent_t t0, t6, t7, t8;
        float t00, t11;
        cudaEventCreate (&t0); // 建立CUDA引發 – 事件開始
        cudaEventCreate (&t6); // 建立CUDA引發 – 事件開始
        cudaEventCreate (&t7); // 建立CUDA引發 – 事件開始
        cudaEventCreate (&t8); // 建立CUDA引發 – 事件開始
        cudaEventRecord(t0, 0); // 記錄事件開始時間結束
        float find_min = 0;
        int rank = 0;
        while(K<2*N-1)
        {            

            cudaEventRecord(t7, 0); // 記錄事件開始時間結束
	    cudaMemcpy(h_UP1D, UP1D, N * N * sizeof(float),cudaMemcpyDeviceToHost);
            reduceSinglePass(size, numThreads, numBlocks, UP1D, dev_d_odata, idx_result, cid, rank); 
	     
            cudaDeviceSynchronize();//等thread都執行完畢
            cudaEventRecord(t8, 0); // 記錄事件結束時間結束
            cudaEventSynchronize(t8);
            cudaEventElapsedTime(&t11, t7, t8); // 計算時間差
            find_min += t11;
            update<<<numBlocks,numThreads>>>(UP1D, idx_result, tmp, N, K, c_node);
            cudaDeviceSynchronize();//等thread都執行完畢
            printf("process: (%d/%d) idx=%d min=%f\n",K,2*N-1,idx_result[0],h_UP1D[idx_result[0]]);
            K++;
        }
        cudaEventRecord(t6, 0); // 記錄事件結束時間結束
        cudaEventSynchronize(t6);
        cudaEventElapsedTime(&t00, t0, t6); // 計算時間差
        
        cudaMemcpy(h_c_node, c_node, 2 * N * sizeof(int),cudaMemcpyDeviceToHost);

        //end = clock();
        for(K=N;K<2*N-1;K++)
        {
                i_min = h_c_node[2*(K-N)];
                j_min = h_c_node[2*(K-N)+1];
                printf("i_min = %3d j_min = %3d\n",i_min,j_min);
                //printf("i_min=%d j_min=%d    indi=%d  indj=%d\n",i_min,j_min,ind[i_min],ind[j_min]);
        
                Build_Tree(node,ind[i_min],ind[j_min],K);
                ind[i_min] = K;
        }
        printf("Find min time = %f\n",t11);
        printf("Update   time = %f\n",t00-t11);
        printf("GPU time = %f\n",t00);
        cudaFree(idx_result);
        cudaFree(dev_d_odata);
        cudaFree(tmp);

        free(h_c_node);
        //free(h_UP1D);
        //free(h_cid);

        cudaFree(cid);
        cudaFree(c_node);        
        cudaFree(UP1D);

}

int main(int argc, char *argv[])
{
        int N;
        const char *filename;
        filename = argv[1];
        //filename = "testcase.txt";
        FILE *fp;
        fp = fopen(filename,"r");
        if(fp==NULL)
        {
                printf("Failed to open file: %s\n", filename);
                return 1;
        }

        //read size
        fscanf(fp,"%d",&N);

        //distribute memory for TREE POINTER
        struct node *node = (struct node*)malloc((2*N-1)*sizeof(struct node));

        //Initialized node
        node_Initial(node,N);
        char *q_tree;
        q_tree = (char*)malloc(N*sizeof(char));
        //FILE *ftree;


        //distribute memory for matrix
        int i,j;

        int *ind;
        //cudaMallocManaged(&ind, N*sizeof(int));
        ind = (int *)malloc( N * sizeof(int));
        //read distance matrix

        float *h_UP1D;
        float *UP1D;
        h_UP1D = (float *)malloc(N * N * sizeof(float));
        cudaMalloc((void**)&UP1D, N * N * sizeof(float));
        //cudaMallocManaged(&UP1D, N * N * sizeof(float));
               
        

        printf("Reading data from file ...\n");
        for(i=0;i<N;i++)
        {
            ind[i]=i;//index
            fscanf(fp,"%s",&q_tree[i]);
            //printf("q_tree = %c \n",q_tree[i]);
            //fprintf(ftree,"%s ",&q_tree[i]);
            for(j=0;j<N;j++)
            {
                fscanf(fp,"%f,",&h_UP1D[i * N + j]);
                //printf("%3.0f ",UPCluster[i][j]);
            }
            //printf("\n");
        }
        cudaMemcpy(UP1D, h_UP1D, N * N * sizeof(float),cudaMemcpyHostToDevice); 

        printf("Read completed\n");
        fclose(fp);

        printf("Source Martix: \n");   
        //show(UPCluster,N);

        printf("-----------------------------\n\n");

        int *h_cid;
        int *cid;
        h_cid = (int *)malloc( N * N * sizeof(int));
        cudaMalloc((void**)&cid, N * N * sizeof(int));
        //cudaMallocManaged(&cid, size * sizeof(int));
        for(i = 0; i < N; i++)
        {
                for(j = 0; j < N; j++)
                {
                        h_cid[i * N + j] = i * N + j;
                        
                }
        }
        cudaMemcpy(cid, h_cid, N * N * sizeof(int),cudaMemcpyHostToDevice);
        free(h_cid); 

        int K=N;
        printf("K=%d N=%d\n",K,N);

        printf("Start\n");
        cpu(K,N,UP1D,ind,node,cid,h_UP1D);
        printf("Start preorder\n");
        preorder(&node[2*N-2],N);
        //FILE *fw;
        //fw = fopen("up_result.txt","w");
        printf("Start Inoder_Result\n");
        Inoder_Result(&node[2*N-2],N);
        printf("End\n");
        free(h_UP1D); 
        free(q_tree); 
        free(node);
        cudaFree(ind);
        cudaFree(UP1D);
        cudaFree(cid);
}
