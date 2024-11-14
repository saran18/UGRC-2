#include <bits/stdc++.h>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std;

const int MAX_HULL_SIZE = 5000;

__device__ int DEBUG = 0;
int HDEBUG = 0;

struct Point
{
public:
    double x;
    double y;
    __host__ __device__ Point() : x(0), y(0) {};
    __host__ __device__ Point(double x, double y) : x(x), y(y) {};

    __host__ __device__ bool operator==(const Point &other) const
    {
        return (x == other.x) && (y == other.y);
    }

    friend std::ostream &operator<<(std::ostream &os, const Point &point)
    {
        os << "(" << point.x << ", " << point.y << ")";
        return os;
    }
};

__host__ __device__ double distance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

__host__ __device__ int orientation(const Point &p, const Point &q, const Point &r)
{
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    const double epsilon = 1e-9;
    if (fabs(val) < epsilon)
    {
        // Collinear points: switched the 2,1 (if r is farther than q from p => q should be excluded from the hull)
        if (distance(p, q) < distance(p, r))
            return 2;
        else if (distance(p, q) > distance(p, r))
            return 1;
        else
            return 0;
    }
    return (val < 0) ? 1 : 2;
}

__device__ bool compare(const Point &p1, const Point &p2, const Point &base)
{
    int o = orientation(base, p1, p2);
    if (o == 0)
    {
        return distance(base, p1) < distance(base, p2);
    }
    return o == 2;
}

__global__ void graham_scan_kernel(Point *points, Point *hulls, int *hull_sizes, int group_size, int num_points, int k)
{
    int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= k)
        return;
    int start_idx = group_id * group_size;
    int end_idx = min(start_idx + group_size, num_points);
    if (DEBUG && group_id == 0)
    {
        printf("Group %d: start_idx: %d, end_idx: %d\n", group_id, start_idx, end_idx);
        for (int i = start_idx; i < end_idx; i++)
        {
            printf("(%f, %f) ", points[i].x, points[i].y);
        }
        printf("\n");
    }
    Point base = points[start_idx];
    int base_index = start_idx;
    for (int i = start_idx; i < end_idx; i++)
    {
        if (points[i].y < base.y ||
            (points[i].y == base.y && points[i].x < base.x))
        {
            base = points[i];
            base_index = i;
        }
    }
    points[base_index] = points[start_idx];
    points[start_idx] = base;
    for (int i = start_idx + 1; i < end_idx; i++)
    {
        for (int j = i + 1; j < end_idx; j++)
        {
            if (compare(points[i], points[j], base))
            {
                Point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
    int curGroupSize = end_idx - start_idx;
    if (curGroupSize < 3)
    {
        hull_sizes[group_id] = curGroupSize;
        for (int i = start_idx; i < end_idx; i++)
        {
            hulls[group_id * MAX_HULL_SIZE + i - start_idx] = points[i];
        }
        return;
    }
    Point hull[MAX_HULL_SIZE];
    int hull_count = 0;
    hull[hull_count++] = points[start_idx];
    hull[hull_count++] = points[start_idx + 1];
    hull[hull_count++] = points[start_idx + 2];
    for (int i = start_idx + 3; i < end_idx; i++)
    {
        while (hull_count >= 2 && orientation(hull[hull_count - 2], hull[hull_count - 1], points[i]) != 1)
        {
            hull_count--;
        }
        hull[hull_count++] = points[i];
    }
    hull_sizes[group_id] = hull_count;
    for (size_t i = 0; i < hull_count; i++)
    {
        hulls[group_id * MAX_HULL_SIZE + i] = hull[i];
    }
}

void printPerfStats(std::chrono::time_point<std::chrono::high_resolution_clock> startSort,
                    std::chrono::time_point<std::chrono::high_resolution_clock> endSort,
                    std::chrono::time_point<std::chrono::high_resolution_clock> startTransferToDevice,
                    std::chrono::time_point<std::chrono::high_resolution_clock> endTransferToDevice,
                    std::chrono::time_point<std::chrono::high_resolution_clock> startKernel,
                    std::chrono::time_point<std::chrono::high_resolution_clock> endKernel,
                    std::chrono::time_point<std::chrono::high_resolution_clock> startTransferToHost,
                    std::chrono::time_point<std::chrono::high_resolution_clock> endTransferToHost)
{
    auto sortTime = std::chrono::duration<double>(endSort - startSort);
    auto transferToDeviceTime = std::chrono::duration<double>(endTransferToDevice - startTransferToDevice);
    auto kernelTime = std::chrono::duration<double>(endKernel - startKernel);
    auto transferToHostTime = std::chrono::duration<double>(endTransferToHost - startTransferToHost);
    cout << "Sort time: " << sortTime.count() << " seconds" << endl;
    cout << "Transfer to device time: " << transferToDeviceTime.count() << " seconds" << endl;
    cout << "Kernel time: " << kernelTime.count() << " seconds" << endl;
    cout << "Transfer to host time: " << transferToHostTime.count() << " seconds" << endl;
    auto totalTime = sortTime + transferToDeviceTime + kernelTime + transferToHostTime;
    cout << "Total time: " << totalTime.count() << " seconds" << endl;
}

__host__ int findRightmostPoint(const vector<Point> &hull, int start, int size)
{
    int rightmost = start;
    for (int i = start; i < start + size; i++)
    {
        if (hull[i].x > hull[rightmost].x ||
            (hull[i].x == hull[rightmost].x && hull[i].y < hull[rightmost].y))
        {
            rightmost = i;
        }
    }
    return rightmost - start;
}

__host__ int findLeftmostPoint(const vector<Point> &hull, int start, int size)
{
    int leftmost = start;
    for (int i = start; i < start + size; i++)
    {
        if (hull[i].x < hull[leftmost].x ||
            (hull[i].x == hull[leftmost].x && hull[i].y < hull[leftmost].y))
        {
            leftmost = i;
        }
    }
    return leftmost - start;
}

__host__ int nextIndex(int idx, int size, bool antiClockwise)
{
    if (antiClockwise)
    {
        return (idx + 1) % size;
    }
    return (idx - 1 + size) % size;
}

__host__ void findTangents(const vector<Point> &hull1, int size1,
                           const vector<Point> &hull2, int start2, int size2,
                           int &upper1, int &upper2,
                           int &lower1, int &lower2)
{
    lower1 = upper1 = findRightmostPoint(hull1, 0, size1);
    lower2 = upper2 = findLeftmostPoint(hull2, start2, size2);
    if (HDEBUG)
    {
        cout << "Rightmost point of hull1: " << hull1[lower1] << endl;
        cout << "Leftmost point of hull2: " << hull2[start2 + lower2] << endl;
    }

    // TODO:  HANDLE !!!!
    // bool isVerticalAligned = (hull1[lower1].x == hull2[start2 + lower2].x);

    bool changed;
    do
    {
        changed = false;
        // new p1 in CW => convex hull set (previous element)
        // new p2 in CCW => convex hull set (next element)
        int next1 = nextIndex(lower1, size1, false);
        int next2 = nextIndex(lower2, size2, true);

        if (orientation(hull2[start2 + lower2], hull1[lower1], hull1[next1]) == 1)
        {
            lower1 = next1;
            changed = true;
        }
        if (orientation(hull1[lower1], hull2[start2 + lower2], hull2[start2 + next2]) == 2)
        {
            lower2 = next2;
            changed = true;
        }
    } while (changed);

    do
    {
        changed = false;
        int next1 = nextIndex(upper1, size1, true);
        int next2 = nextIndex(upper2, size2, false);
        if (orientation(hull2[start2 + upper2], hull1[upper1], hull1[next1]) == 2)
        {
            upper1 = next1;
            changed = true;
        }
        if (orientation(hull1[upper1], hull2[start2 + upper2], hull2[start2 + next2]) == 1)
        {
            upper2 = next2;
            changed = true;
        }
    } while (changed);
}

__host__ vector<Point> mergeTwoHulls(const vector<Point> &hull1, int size1,
                                     const vector<Point> &hull2, int start2, int size2)
{
    // Handle empty hulls
    if (size1 == 0)
    {
        return vector<Point>(hull2.begin() + start2, hull2.begin() + start2 + size2);
    }
    if (size2 == 0)
    {
        return vector<Point>(hull1.begin(), hull1.begin() + size1);
    }

    if (HDEBUG)
    {
        cout << "\nMerging hulls:" << endl;
        cout << "Hull1 size: " << size1 << ", points: ";
        for (int i = 0; i < size1; i++)
            cout << hull1[i] << " ";
        cout << "\nHull2 size: " << size2 << ", points: ";
        for (int i = 0; i < size2; i++)
            cout << hull2[start2 + i] << " ";
        cout << endl;
    }

    int upper1, upper2, lower1, lower2;
    findTangents(hull1, size1, hull2, start2, size2, upper1, upper2, lower1, lower2);

    if (HDEBUG)
    {
        cout << "Tangents: " << endl;
        cout << "Upper tangent: " << hull1[upper1] << " - " << hull2[start2 + upper2] << endl;
        cout << "Lower tangent: " << hull1[lower1] << " - " << hull2[start2 + lower2] << endl;
    }

    vector<Point> merged;
    merged.reserve(size1 + size2); // Pre-allocate space

    // Add points from hull1, from upper1 go CCW until lower1
    int idx = upper1;
    do
    {
        merged.push_back(hull1[idx]);
        idx = (idx + 1) % size1;
    } while (idx != lower1);
    merged.push_back(hull1[lower1]);

    // Add points from hull2 (lower tangent to upper tangent)
    idx = lower2;
    do
    {
        merged.push_back(hull2[start2 + idx]);
        idx = (idx + 1) % size2;
    } while (idx != upper2);
    merged.push_back(hull2[start2 + upper2]);

    if (HDEBUG)
    {
        cout << "Merged hull size: " << merged.size() << ", points: ";
        for (const Point &p : merged)
            cout << p << " ";
        cout << endl;
    }

    return merged;
}

vector<Point> mergeAllHulls(const vector<Point> &hulls, const vector<int> &hull_sizes, int k)
{
    if (k == 1)
    {
        return vector<Point>(hulls.begin(), hulls.begin() + hull_sizes[0]);
    }

    if (HDEBUG)
    {
        cout << "Starting merge of " << k << " hulls" << endl;
    }

    // First merge: Create initial merged hull
    vector<Point> result(hulls.begin(), hulls.begin() + hull_sizes[0]);

    // Iteratively merge remaining hulls
    for (int i = 1; i < k; i++)
    {
        if (HDEBUG)
        {
            cout << "\nMerging hull " << i << endl;
        }
        result = mergeTwoHulls(result, result.size(), hulls, i * MAX_HULL_SIZE, hull_sizes[i]);
    }

    return result;
}

void computeConvexHull(vector<Point> &points, int n, int k)
{
    int group_size = (n + k - 1) / k;
    auto startSort = std::chrono::high_resolution_clock::now();
    thrust::device_vector<Point> thrust_d_points(points.begin(), points.end());
    thrust::sort(thrust_d_points.begin(), thrust_d_points.end(), [] __device__(const Point &p1, const Point &p2)
                 { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });
    thrust::copy(thrust_d_points.begin(), thrust_d_points.end(), points.begin());
    auto endSort = std::chrono::high_resolution_clock::now();

    auto startTransferToDevice = std::chrono::high_resolution_clock::now();
    Point *d_points;
    hipMalloc(&d_points, sizeof(Point) * n);
    hipMemcpy(d_points, points.data(), sizeof(Point) * n, hipMemcpyHostToDevice);
    Point *d_hulls;
    hipMalloc(&d_hulls, sizeof(Point) * k * MAX_HULL_SIZE);
    int *d_hull_sizes;
    hipMalloc(&d_hull_sizes, sizeof(int) * k);
    auto endTransferToDevice = std::chrono::high_resolution_clock::now();

    const int threadsPerBlock = 256;
    const int blocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    auto startKernel = std::chrono::high_resolution_clock::now();
    graham_scan_kernel<<<blocks, threadsPerBlock>>>(d_points, d_hulls, d_hull_sizes, group_size, n, k);
    hipDeviceSynchronize();
    auto endKernel = std::chrono::high_resolution_clock::now();

    auto startTransferToHost = std::chrono::high_resolution_clock::now();
    vector<Point> hulls(k * MAX_HULL_SIZE);
    vector<int> hull_sizes(k);
    hipMemcpy(hulls.data(), d_hulls, sizeof(Point) * k * MAX_HULL_SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(hull_sizes.data(), d_hull_sizes, sizeof(int) * k, hipMemcpyDeviceToHost);
    auto endTransferToHost = std::chrono::high_resolution_clock::now();
    if (HDEBUG)
    {
        for (int i = 0; i < k; i++)
        {
            cout << "Convex Hull " << i << ": ";
            for (int j = 0; j < hull_sizes[i]; j++)
            {
                cout << hulls[i * MAX_HULL_SIZE + j] << " ";
            }
            cout << endl;
        }
    }
    std::cout << "Individual Convex Hulls computed" << std::endl;

    // -------- Merge Convex Hulls --------
    auto startMerge = std::chrono::high_resolution_clock::now();
    vector<Point> final_hull = mergeAllHulls(hulls, hull_sizes, k);
    auto endMerge = std::chrono::high_resolution_clock::now();
    cout << "Merge time : " << std::chrono::duration<double>(endMerge - startMerge).count() << endl;
    if (HDEBUG)
    {
        cout << "Final merged hull size: " << final_hull.size() << endl;
        for (const Point &p : final_hull)
        {
            cout << p << " ";
        }
    }
    printPerfStats(startSort, endSort, startTransferToDevice, endTransferToDevice, startKernel, endKernel, startTransferToHost, endTransferToHost);
}

int main(int argc, char **argv)
{
    string inputFileName = argv[1];
    int k = stoi(argv[2]);
    printf("k: %d\n", k);
    ifstream fin(inputFileName);
    if (!fin.is_open())
    {
        cout << "Error opening file" << endl;
        return 1;
    }
    int n;
    fin >> n;
    printf("n: %d\n", n);
    vector<Point> points(n);
    for (int i = 0; i < n; i++)
    {
        double x, y;
        fin >> x >> y;
        points[i] = Point(x, y);
    }
    computeConvexHull(points, n, k);
    return 0;
}