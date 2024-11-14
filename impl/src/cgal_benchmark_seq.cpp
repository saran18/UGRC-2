#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <vector>
#include <iostream>

// Define CGAL Kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;

int main()
{
    // Define the set of points
    std::vector<Point_2> points = {
        Point_2(0, 0), Point_2(1, 1), Point_2(2, 2), Point_2(3, 1),
        Point_2(4, 0), Point_2(2, -1), Point_2(1, -2), Point_2(3, -2),
        Point_2(-1, 0), Point_2(-2, 1), Point_2(-3, 2), Point_2(-1, -1),
        Point_2(-2, -2), Point_2(0, 3), Point_2(3, 3), Point_2(-3, -3)
    };

    // Vector to store the resulting convex hull points
    std::vector<Point_2> result;

    // Compute convex hull
    CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));

    // Print the points on the convex hull
    std::cout << "Convex Hull:" << std::endl;
    for (const auto& point : result)
    {
        std::cout << "(" << point.x() << ", " << point.y() << ")" << std::endl;
    }

    return 0;
}
