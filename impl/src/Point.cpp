#include "Point.h"
#include <cmath>
#include <hip/hip_runtime.h>

// Default constructor
__host__ __device__ Point::Point() : x(0), y(0) {}

// Parameterized constructor
__host__ __device__ Point::Point(double x, double y) : x(x), y(y) {}

// Getter for x coordinate
__host__ __device__ double Point::getX() const {
    return x;
}

// Getter for y coordinate
__host__ __device__ double Point::getY() const {
    return y;
}

// Setter for x coordinate
__host__ __device__ void Point::setX(double x) {
    this->x = x;
}

// Setter for y coordinate
__host__ __device__ void Point::setY(double y) {
    this->y = y;
}

// Compute distance to another Point
__host__ __device__ double Point::distanceTo(const Point &other) const {
    double dx = other.x - x;
    double dy = other.y - y;
    return sqrt(dx * dx + dy * dy);
}

std::ostream& operator<<(std::ostream& os, const Point& point)
{
    os << "(" << point.x << ", " << point.y << ")";
    return os;
}
