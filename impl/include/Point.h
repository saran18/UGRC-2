#ifndef POINT_H
#define POINT_H

struct Point
{
public:
    double x;
    double y;
    __host__ __device__ Point();
    __host__ __device__ Point(double x, double y);
    __host__ __device__ double getX() const;
    __host__ __device__ double getY() const;
    __host__ __device__ void setX(double x);
    __host__ __device__ void setY(double y);
    __host__ __device__ double distanceTo(const Point &other) const;

    friend std::ostream& operator<<(std::ostream& os, const Point& point);
};

#endif // POINT_H
