#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
typedef struct node *TREE_ptr;
struct node
{
        int NODE;
        int LEVEL;
        TREE_ptr ancestor, left_child, right_child;
        int left_level, right_level;
};

void del(int N, int i_min,int j_min, float **UPCluster)
{
        int j;
        for(j=0;j<N;j++)
        {
                UPCluster[j_min][j]=0;
                UPCluster[i_min][j]=0;
                UPCluster[j][j_min]=0;
                UPCluster[j][i_min]=0;
        }
}
void replace(int N, int i_min, int j_min, float **UPCluster)
{
        int j;
        //replace i,j with K
        for(j=0;j<N;j++)
        {
                UPCluster[i_min][j] = UPCluster[N][j];
                UPCluster[j][i_min] = UPCluster[i_min][j];
        }
}

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

void cpu(int N,int K,float **UPCluster,int *ind,struct node *node)
{
        int i=0,j=0;
        //int size = N*N*sizeof(float);
        //save node
        int *c_node;
        c_node = (int*)malloc(2*N*sizeof(int));

        int i_min,j_min;
        //float min;
    cudaEvent_t t0, t1;
        float t00;
        cudaEventCreate (&t0); // 建立CUDA引發 – 事件開始
    cudaEventCreate (&t1); // 建立CUDA引發 – 事件開始
    cudaEventRecord(t0, 0); // 記錄事件開始時間結束
        //clock_t start,end;
        //start = clock();
        while(K<2*N-1)
        {
        //printf("K=%d N=%d\n",K,N);
        //每個ROW找0之前的最小值，並記錄其位置
                float min = 1000;
                for(i=0;i<N;i++)
                {
                        for(j=0;j<i;j++)
                        {
                                if((UPCluster[i][j] < min) && (UPCluster[i][j] != 0))
                                {
                                        i_min = i;
                                        j_min = j;
                                        min = UPCluster[i][j];
                                }
                        }
                }
                
                c_node[2*(K-N)] = i_min;
                c_node[2*(K-N)+1] = j_min;

                for(j=0;j<N;j++)
                {
                        UPCluster[N][j] = (UPCluster[i_min][j] + UPCluster[j_min][j])/2.0;
                }
                UPCluster[N][i_min] = UPCluster[N][j_min] = 0;

                //delete i j
                del(N,i_min,j_min,UPCluster);

                //replace i,j with K
                replace(N,i_min,j_min,UPCluster);
                //int q = 0;
                //scanf("%d",&q);
                //show(UPCluster,N);
                printf("min = %5.2f i=%d j=%d\n",min,i_min,j_min);
                //printf("%d\n",K);
                K++;
        }
        cudaThreadSynchronize();//等thread都執行完畢
        cudaEventRecord(t1, 0); // 記錄事件結束時間結束
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&t00, t0, t1); // 計算時間差

        //end = clock();
        for(K=N;K<2*N-1;K++)
        {
                i_min = c_node[2*(K-N)];
                j_min = c_node[2*(K-N)+1];
                //printf("i_min = %3d j_min = %3d\n",i_min,j_min);
        
                Build_Tree(node,ind[i_min],ind[j_min],K);
                ind[i_min] = K;
        }
        printf("CPU time = %f\n",t00);
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
        float **UPCluster;
        int *ind;
        UPCluster = (float **)malloc((N+1)*sizeof(float*));
        for(i=0;i<(N+1);i++)
        {
                UPCluster[i] =(float*)malloc(N*sizeof(float*));
        }
        ind = (int *)malloc(N*sizeof(int));
        //read distance matrix
        do
        {
            for(i=0;i<N;i++)
            {
                ind[i]=i;//index
                fscanf(fp,"%s",&q_tree[i]);
                //printf("q_tree = %c \n",q_tree[i]);
                //fprintf(ftree,"%s ",&q_tree[i]);
                for(j=0;j<N;j++)
                {
                    fscanf(fp,"%f,",&UPCluster[i][j]);
                    //printf("%3.0f ",UPCluster[i][j]);
                }
                //printf("\n");
            }
        }while(fscanf(fp,"%f",UPCluster[i])!=EOF);
        
        fclose(fp);
        printf("Source : \n");

            //show(UPCluster,N);

        printf("-----------------------------\n\n");
        int K=N;
        printf("K=%d N=%d\n",K,N);

        printf("Start\n\n\n");
        cpu(K,N,UPCluster,ind,node);
        preorder(&node[2*N-2],N);
        //FILE *fw;
        //fw = fopen("up_result.txt","w");
        Inoder_Result(&node[2*N-2],N);

        //fclose(fw);
        //fclose(ftree);
        printf("\n");
        free(q_tree);
        free(UPCluster);
        free(node);
        free(ind);
}
