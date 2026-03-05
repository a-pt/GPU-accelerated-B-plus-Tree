
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
using namespace std;
const int MAXI = 7;

class Managed {
public:
    void* operator new(size_t len) {
        void* ptr;
        cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

// BP node
class Node : public Managed {
    bool IS_LEAF;
    int* key, size;
    Node** ptr;
    friend class BPTree;

public:
    __host__ Node();
};

// BP tree
class BPTree : public Managed {
    Node* root;
    __host__ void insertInternal(int, Node*, Node*);
    __host__ __device__ Node* findParent(Node*, Node*);

public:
    __host__ BPTree();
    __host__ void init() { root = NULL; }  // safe init after cudaHostAlloc (raw memory, no constructor)
    __host__ __device__ Node* search(int);
    __host__  void insert(int, int*);
    __host__ __device__ void display(Node*, int);
    __host__ __device__ int rangequery(int, int, int**);
    __host__ __device__ int pathtrace(int, int*);
    __host__ __device__ void addition(int, int, int);
    __host__ __device__ int height();
    __host__ __device__ Node* getRoot();
};

// Constructor of Node
Node::Node()
{
    cudaHostAlloc(&key, sizeof(int) * MAXI, cudaHostAllocDefault);
    cudaHostAlloc(&ptr, sizeof(Node*) * (MAXI + 1), cudaHostAllocDefault);
    IS_LEAF = false;  // initialize to safe default
    size = 0;         // initialize to safe default
}

// Initialise the BPTree Node
BPTree::BPTree()
{
    root = NULL;
}

// Function to find any element
// in B+ Tree
Node* BPTree::search(int x)
{

    // If tree is empty
    if (root == NULL) {
        printf("Tree is empty\n");
        return NULL;
    }

    // Traverse to find the value
    else {

        Node* cursor = root;

        // Till we reach leaf node
        while (cursor->IS_LEAF == false) {

            for (int i = 0;
                i < cursor->size; i++) {

                // If the element to be
                // found is not present
                if (x < cursor->key[i]) {
                    cursor = cursor->ptr[i];
                    break;
                }

                // If reaches end of the
                // cursor node
                if (i == cursor->size - 1) {
                    cursor = cursor->ptr[i + 1];
                    break;
                }
            }
        }

        // Traverse the cursor and find
        // the node with value x
        for (int i = 0;
            i < cursor->size; i++) {

            // If found then return
            if (cursor->key[i] == x) {
                //cout << "Found\n";
                return cursor->ptr[i];
            }
        }

        // Else element is not present
        return NULL;
    }
}

// Function to implement the Insert Operation in B+ Tree
void BPTree::insert(int x, int* record)
{

    // If root is null then return newly created node
    if (root == NULL) {
        root = new Node;
        root->key[0] = x;
        root->ptr[0] = (Node*)record;
        root->ptr[1] = NULL;
        root->IS_LEAF = true;
        root->size = 1;
    }

    // Traverse the B+ Tree
    else {
        Node* cursor = root;
        Node* parent;

        // Till cursor reaches the leaf node
        while (cursor->IS_LEAF == false) {

            parent = cursor;

            for (int i = 0; i < cursor->size; i++) {

                // If found the position where we have to insert node
                if (x < cursor->key[i]) {
                    cursor = cursor->ptr[i];
                    break;
                }

                // If reaches the end
                if (i == cursor->size - 1) {
                    cursor = cursor->ptr[i + 1];
                    break;
                }
            }
        }

        if (cursor->size < MAXI) {

            int i = 0;
            while (x > cursor->key[i]
                && i < cursor->size) {
                i++;
            }

            for (int j = cursor->size; j > i; j--) {
                cursor->key[j] = cursor->key[j - 1];
            }
            cursor->key[i] = x;

            for (int j = cursor->size + 1; j > i; j--) {
                cursor->ptr[j] = cursor->ptr[j - 1];
            }
            cursor->ptr[i] = (Node*)record;
            cursor->size++;
        }

        else {
            // Create a newLeaf node
            Node* newLeaf = new Node;
            //Node* newLeaf;
            int virtualNode[MAXI + 1];
            Node* virtualPtr[MAXI + 2];
            // Update cursor to virtual
            // node created
            for (int i = 0; i < MAXI; i++) {
                virtualNode[i]
                    = cursor->key[i];
            }
            for (int i = 0; i < MAXI + 1; i++) {
                virtualPtr[i] = cursor->ptr[i];
            }
            int i = 0, j;

            // Traverse to find where the new
            // node is to be inserted
            while (x > virtualNode[i] && i < MAXI) {
                i++;
            }

            // Update the current virtual
            // Node to its previous
            for (int j = MAXI; j > i; j--) {
                virtualNode[j] = virtualNode[j - 1];
            }
            virtualNode[i] = x;

            for (int j = MAXI + 1; j > i; j--) {
                virtualPtr[j] = virtualPtr[j - 1];
            }
            virtualPtr[i] = (Node*)record;

            newLeaf->IS_LEAF = true;

            cursor->size = (MAXI + 1) / 2;
            newLeaf->size = MAXI + 1 - (MAXI + 1) / 2;

            cursor->ptr[cursor->size] = newLeaf;

            //newLeaf->ptr[newLeaf->size] = cursor->ptr[MAXI];
            newLeaf->ptr[newLeaf->size] = virtualPtr[MAXI + 1];
            cursor->ptr[MAXI] = NULL;

            // Update the current virtual
            // Node's key to its previous
            for (i = 0; i < cursor->size; i++) {
                cursor->key[i] = virtualNode[i];
                cursor->ptr[i] = virtualPtr[i];
            }


            // Update the newLeaf key to
            // virtual Node
            for (i = 0, j = cursor->size; i < newLeaf->size; i++, j++) {
                newLeaf->key[i] = virtualNode[j];
                newLeaf->ptr[i] = virtualPtr[j];
            }

            // If cursor is the root node
            if (cursor == root) {

                // Create a new Node
                Node* newRoot = new Node;
                //Node* newRoot;
                // Update rest field of
                // B+ Tree Node
                newRoot->key[0] = newLeaf->key[0];
                newRoot->ptr[0] = cursor;
                newRoot->ptr[1] = newLeaf;
                newRoot->IS_LEAF = false;
                newRoot->size = 1;
                root = newRoot;
            }
            else {
                // Recursive Call for insert in internal
                insertInternal(newLeaf->key[0], parent, newLeaf);
            }
        }
    }
}

// Function to implement the Insert
// Internal Operation in B+ Tree
void BPTree::insertInternal(int x, Node* cursor, Node* child)
{

    // If we doesn't have overflow
    if (cursor->size < MAXI) {
        int i = 0;

        // Traverse the child node for current cursor node
        while (x > cursor->key[i] && i < cursor->size) {
            i++;
        }

        // Traverse the cursor node and update the current key to its previous node key
        for (int j = cursor->size; j > i; j--) {
            cursor->key[j] = cursor->key[j - 1];
        }

        // Traverse the cursor node and update the current ptr to its previous node ptr
        for (int j = cursor->size + 1; j > i + 1; j--) {
            cursor->ptr[j] = cursor->ptr[j - 1];
        }

        cursor->key[i] = x;
        cursor->size++;
        cursor->ptr[i + 1] = child;
    }

    // For overflow, break the node
    else {

        // For new Interval
        Node* newInternal = new Node;
        //Node* newInternal;
        int virtualKey[MAXI + 1];
        Node* virtualPtr[MAXI + 2];

        // Insert the current list key of cursor node to virtualKey
        for (int i = 0; i < MAXI; i++) {
            virtualKey[i] = cursor->key[i];
        }

        // Insert the current list ptr of cursor node to virtualPtr
        for (int i = 0; i < MAXI + 1; i++) {
            virtualPtr[i] = cursor->ptr[i];
        }

        int i = 0, j;

        // Traverse to find where the new node is to be inserted
        while (x > virtualKey[i] && i < MAXI) {
            i++;
        }

        // Traverse the virtualKey node and update the current key to its previous node key
        for (int j = MAXI; j > i; j--) {
            virtualKey[j] = virtualKey[j - 1];
        }

        virtualKey[i] = x;

        // Traverse the virtualKey node and update the current ptr to its previous node ptr
        for (int j = MAXI + 1; j > i + 1; j--) {
            virtualPtr[j] = virtualPtr[j - 1];
        }

        virtualPtr[i + 1] = child;
        newInternal->IS_LEAF = false;

        cursor->size = (MAXI + 1) / 2;

        newInternal->size = MAXI - (MAXI + 1) / 2;

        for (i = 0; i < cursor->size; i++) {
            cursor->key[i] = virtualKey[i];
        }

        for (i = 0; i < cursor->size + 1; i++) {
            cursor->ptr[i] = virtualPtr[i];
        }
        // Insert new node as an internal node
        for (i = 0, j = cursor->size + 1; i < newInternal->size; i++, j++) {
            newInternal->key[i] = virtualKey[j];
        }

        for (i = 0, j = cursor->size + 1; i < newInternal->size + 1; i++, j++) {
            newInternal->ptr[i] = virtualPtr[j];
        }

        // If cursor is the root node
        if (cursor == root) {

            // Create a new root node
            Node* newRoot = new Node;
            //Node* newRoot;
            // Update key value
           // newRoot->key[0] = cursor->key[cursor->size];
            newRoot->key[0] = virtualKey[cursor->size];
            // Update rest field of B+ Tree Node
            newRoot->ptr[0] = cursor;
            newRoot->ptr[1] = newInternal;
            newRoot->IS_LEAF = false;
            newRoot->size = 1;
            root = newRoot;
        }

        else {
            // Recursive Call to insert the data
            insertInternal(virtualKey[cursor->size], findParent(root, cursor), newInternal);
        }
    }
}

// Function to find the parent node
Node* BPTree::findParent(Node* cursor, Node* child)
{
    Node* parent;

    // If cursor reaches the end of Tree
    if (cursor->IS_LEAF || (cursor->ptr[0])->IS_LEAF) {
        return NULL;
    }

    // Traverse the current node with all its child
    for (int i = 0; i < cursor->size + 1; i++) {

        // Update the parent for the child Node
        if (cursor->ptr[i] == child) {
            parent = cursor;
            return parent;
        }

        // Else recursively traverse to find child node
        else {
            parent = findParent(cursor->ptr[i], child);

            // If parent is found, then
            // return that parent node
            if (parent != NULL)
                return parent;
        }
    }

    // Return parent node
    return parent;
}

// Function to get the root Node
Node* BPTree::getRoot()
{
    return root;
}

void BPTree::display(Node* cursor, int level)
{
    printf("Level %d: ", level);
    for (int i = 0; i < cursor->size; i++)
    {
        printf("%d-", cursor->key[i]);
    }
    printf("\n");
    if (cursor->IS_LEAF)
        return;
    else
    {
        for (int i = 0; i < cursor->size + 1; i++)
        {
            display(cursor->ptr[i], level + 1);
        }
    }
}

int BPTree::rangequery(int a, int b, int** res)
{
    int z = 0;
    if (root == NULL) {
        printf("Tree is empty\n");
        return -1;
    }

    // Traverse to find the value
    else {

        Node* cursor = root;

        // Till we reach leaf node
        while (cursor->IS_LEAF == false) {

            for (int i = 0;
                i < cursor->size; i++) {

                // If the element to be
                // found is not present
                if (a < cursor->key[i]) {
                    cursor = cursor->ptr[i];
                    break;
                }

                // If reaches end of the
                // cursor node
                if (i == cursor->size - 1) {
                    cursor = cursor->ptr[i + 1];
                    break;
                }
            }
        }

        // Traverse the cursor and find
        // the node with value x
        while (cursor->key[0] <= b) {
            for (int i = 0;
                i < cursor->size; i++) {

                // If found then return
                if (cursor->key[i] >= a && cursor->key[i] <= b) {
                    res[z] = (int*)cursor->ptr[i];
                    z++;
                }
            }
            cursor = cursor->ptr[cursor->size];
            if (cursor == NULL)
                break;
        }
        return z;
    }
}

int BPTree::pathtrace(int x, int* res)
{
    int index = 0;
    // If tree is empty
    if (root == NULL) {
        printf("Tree is empty\n");
        return -1;
    }

    // Traverse to find the value
    else {

        Node* cursor = root;

        // Till we reach leaf node
        while (cursor->IS_LEAF == false) {
            res[index] = cursor->key[0];
            index++;
            for (int i = 0; i < cursor->size; i++) {

                // If the element to be
                // found is not present
                if (x < cursor->key[i]) {
                    cursor = cursor->ptr[i];
                    break;
                }

                // If reaches end of the
                // cursor node
                if (i == cursor->size - 1) {
                    cursor = cursor->ptr[i + 1];
                    break;
                }
            }
        }
        res[index] = cursor->key[0];
        index++;
        return index;
    }
}

void BPTree::addition(int x, int att, int val)
{

    // If tree is empty
    if (root == NULL) {
        printf("Tree is empty\n");
    }

    // Traverse to find the value
    else {

        Node* cursor = root;

        // Till we reach leaf node
        while (cursor->IS_LEAF == false) {

            for (int i = 0;
                i < cursor->size; i++) {

                // If the element to be
                // found is not present
                if (x < cursor->key[i]) {
                    cursor = cursor->ptr[i];
                    break;
                }

                // If reaches end of the
                // cursor node
                if (i == cursor->size - 1) {
                    cursor = cursor->ptr[i + 1];
                    break;
                }
            }
        }

        // Traverse the cursor and find
        // the node with value x
        for (int i = 0;
            i < cursor->size; i++) {

            // If found then return
            if (cursor->key[i] == x) {
                ((int*)cursor->ptr[i])[att] += val;
                return;
            }
        }
    }
}

int BPTree::height()
{
    int h = 1;
    Node* cursor = root;
    if (root == NULL)
        return 0;
    while (cursor->IS_LEAF == false) {
        h++;
        cursor = cursor->ptr[0];
    }
    return h;
}

__global__ void Search(int p, BPTree* node, int* key, int** ptr)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < p)
    {
        int* rs = (int*)((*node).search(key[id]));
        ptr[id] = rs;
    }
}

__global__ void Height(BPTree* node, int* ht)
{
    *ht = (*node).height();
}

__global__ void RangeQuery(int p, int* A, int* B, int* sz, int* z, BPTree* node, int** res)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < p)
    {
        z[id] = (*node).rangequery(A[id], B[id], &res[sz[id]]);
    }
}




// Driver Code
int main(int argc, char** argv)
{
    BPTree* node;
    cudaHostAlloc(&node, sizeof(BPTree), cudaHostAllocDefault);
    node->init();
    cudaDeviceSynchronize();
    int n, m, q;
    //Input file pointer declaration
    FILE* inputfilepointer;

    //File Opening for read
    char* inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    //Checking if file ptr is NULL
    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }


    fscanf(inputfilepointer, "%d", &n);      //scaning for number of records in database
    fscanf(inputfilepointer, "%d", &m);      //scaning for number of attributes
    //cout << n << m;

    int* db;
    cudaHostAlloc(&db, sizeof(int) * n * m, cudaHostAllocDefault);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            int value;
            fscanf(inputfilepointer, "%d", &value);
            db[i * m + j] = value;
        }
    }

    for (int i = 0; i < n; i++)
    {
        (*node).insert(db[i * m], &db[i * m]);
        // cout << "inserted " << i << "\n";
    }
    //DisplayKernel <<<1, 1 >>> (node);

    fscanf(inputfilepointer, "%d", &q); //scan number of operations
    int op, p, k, att, val, a, b;

    char* outputfilename = argv[2];
    FILE* outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");


    for (int i = 0; i < q; i++)
    {
        fscanf(inputfilepointer, "%d", &op); //scan operation
        if (op == 1)
        {
            //cout << "1 ";
            fscanf(inputfilepointer, "%d", &p); //scan number of search queries
            int** ptr = (int**)malloc(sizeof(int*) * p);
            int* key = (int*)malloc(sizeof(int) * p);
            int** gptr;
            int* gkey;
            cudaMalloc((void**)&gptr, sizeof(int*) * p);
            cudaMalloc((void**)&gkey, sizeof(int) * p);
            for (int j = 0; j < p; j++)
            {
                fscanf(inputfilepointer, "%d", &k);
                key[j] = k;
            }
            cudaMemcpy(gkey, key, sizeof(int) * p, cudaMemcpyHostToDevice);
            int block = 64;
            int grid = ceil((float)((float)p / block));
            Search << < grid, block >> > (p, node, gkey, gptr);
            cudaDeviceSynchronize();
            cudaMemcpy(ptr, gptr, sizeof(int*) * p, cudaMemcpyDeviceToHost);
            for (int j = 0; j < p; j++)
            {
                if (ptr[j] == NULL)
                {
                    fprintf(outputfilepointer, "-1");
                }
                else
                {
                    for (int l = 0; l < m; l++)
                    {
                        //cout << (*ptr)[l] << " ";
                        //fprintf(outputfilepointer, "%d ", (*ptr)[l]);
                        fprintf(outputfilepointer, "%d ", *(ptr[j] + l));
                    }
                }
                fprintf(outputfilepointer, "\n");
            }
            cudaFree(gptr);
            free(ptr);
            cudaFree(gkey);
            free(key);
        }
        else if (op == 2)
        {
            // cout << "2 ";
            fscanf(inputfilepointer, "%d", &p); //scan number of range queries
            int* A = (int*)malloc(sizeof(int) * p);
            int* B = (int*)malloc(sizeof(int) * p);
            int* sz = (int*)malloc(sizeof(int) * p);
            int* z = (int*)malloc(sizeof(int) * p);
            int* gA;
            int* gB;
            int* gz;
            int* gsz;
            int** gres;
            cudaMalloc((void**)&gA, sizeof(int) * p);
            cudaMalloc((void**)&gB, sizeof(int) * p);
            cudaMalloc((void**)&gz, sizeof(int) * p);
            cudaMalloc((void**)&gsz, sizeof(int) * p);
            sz[0] = 0;
            for (int j = 0; j < p; j++)
            {
                fscanf(inputfilepointer, "%d", &a);
                fscanf(inputfilepointer, "%d", &b);
                A[j] = a;
                B[j] = b;
                if (j > 0)
                    sz[j] = sz[j - 1] + (B[j - 1] - A[j - 1] + 1);
            }
            int totsize = sz[p - 1] + (B[p - 1] - A[p - 1] + 1);
            int** res = (int**)malloc(sizeof(int*) * totsize);
            cudaMalloc((void**)&gres, sizeof(int*) * totsize);
            cudaMemcpy(gA, A, sizeof(int) * p, cudaMemcpyHostToDevice);
            cudaMemcpy(gB, B, sizeof(int) * p, cudaMemcpyHostToDevice);
            cudaMemcpy(gsz, sz, sizeof(int) * p, cudaMemcpyHostToDevice);
            int block = 64;
            int grid = ceil((float)((float)p / block));
            RangeQuery << < grid, block >> > (p, gA, gB, gsz, gz, node, gres);
            cudaDeviceSynchronize();
            cudaMemcpy(z, gz, sizeof(int) * p, cudaMemcpyDeviceToHost);
            cudaMemcpy(res, gres, sizeof(int*) * totsize, cudaMemcpyDeviceToHost);

            for (int j = 0; j < p; j++)
            {
                int csz = z[j];
                if (csz == 0)
                    fprintf(outputfilepointer, "-1\n");
                else
                {
                    for (int l = 0; l < csz; l++)
                    {
                        int* initial = res[sz[j] + l];
                        for (int o = 0; o < m; o++)
                        {
                            fprintf(outputfilepointer, "%d ", *(initial + o));
                        }
                        fprintf(outputfilepointer, "\n");
                    }
                }
            }
            cudaFree(gz);
            cudaFree(gres);
            cudaFree(gA);
            cudaFree(gB);
            cudaFree(gsz);
            free(sz);
            free(A);
            free(B);
            free(z);
            free(res);
        }
        else if (op == 3)
        {
            // cout << "3 ";
            fscanf(inputfilepointer, "%d", &p); //scan number of addition queries
            int* Key = (int*)malloc(sizeof(int) * p);
            int* Att = (int*)malloc(sizeof(int) * p);
            int* Val = (int*)malloc(sizeof(int) * p);
            int** res = (int**)malloc(sizeof(int*) * p);
            int* gKey;
            int** gres;
            cudaMalloc((void**)&gKey, sizeof(int) * p);
            cudaMalloc((void**)&gres, sizeof(int*) * p);
            for (int j = 0; j < p; j++)
            {
                fscanf(inputfilepointer, "%d", &k);
                fscanf(inputfilepointer, "%d", &att);
                fscanf(inputfilepointer, "%d", &val);
                Key[j] = k;
                Att[j] = att;
                Val[j] = val;
            }
            cudaMemcpy(gKey, Key, sizeof(int) * p, cudaMemcpyHostToDevice);

            int block = 64;
            int grid = ceil((float)((float)p / block));
            Search << < grid, block >> > (p, node, gKey, gres);
            cudaDeviceSynchronize();
            cudaMemcpy(res, gres, sizeof(int*) * p, cudaMemcpyDeviceToHost);
            for (int j = 0; j < p; j++)
            {
                if (res[j] != NULL)
                    *(res[j] + (Att[j] - 1)) += Val[j];
            }

            cudaFree(gres);
            cudaFree(gKey);
            free(Key);
            free(res);
            free(Att);
            free(Val);
        }
        else
        {
            // cout << "4 ";
            int ht = (*node).height();
            int* out = (int*)malloc(sizeof(int) * ht);
            fscanf(inputfilepointer, "%d", &k); //scan the key for pathtracing
            int lt = (*node).pathtrace(k, out);
            for (int j = 0; j < lt; j++)
            {
                fprintf(outputfilepointer, "%d ", out[j]);
            }
            fprintf(outputfilepointer, "\n");
            free(out);
        }
    }
    //(*node).display((*node).getRoot(), 0);
    fclose(outputfilepointer);
    fclose(inputfilepointer);
    return 0;
}