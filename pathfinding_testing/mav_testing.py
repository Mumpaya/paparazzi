import numpy as np
import matplotlib.pyplot as plt


class polygon2d:
    def __init__(self, vertices: np.ndarray, pos, rot):
        self.vertices = vertices
        self.pos = pos
        self.rot = rot

    def transformed_vertices(self):
        theta = np.radians(self.rot)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return self.vertices @ rotation_matrix.T + self.pos

    @classmethod
    def rectangle(cls, width, height, pos=np.array([0, 0]), rot=0):
        w, h = width / 2, height / 2
        vertices = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
        return cls(vertices, pos, rot)

    @classmethod
    def circle(cls, radius, pos=np.array([0, 0])):
        num_points = 20
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        vertices = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        return cls(vertices, pos, np.random.rand()*360)

    # def area(self):
    #     x = self.vertices[:, 0]
    #     y = self.vertices[:, 1]
    #     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    #
    # def perimeter(self):
    #     return np.sum(np.sqrt(np.sum(np.diff(self.vertices, axis=0) ** 2, axis=1))) + np.sqrt(
    #         np.sum((self.vertices[0] - self.vertices[-1]) ** 2)
    #     )

    def plot(self):
        verts = self.transformed_vertices()
        verts = np.append(verts, [verts[0]], axis=0)  # Close the polygon
        plt.plot(verts[:, 0], verts[:, 1], 'b-')
        plt.fill(verts[:, 0], verts[:, 1], 'cyan', alpha=0.5)
        plt.axis('equal')
        plt.grid()

    def closest_intersection(self, ray_origin, ray_dir):
        depth = float('inf')
        verts = self.transformed_vertices()
        for i in range(len(verts)):
            p1, p2 = verts[i], verts[(i + 1) % len(verts)]
            edge_vec = p2 - p1
            edge_normal = np.array([-edge_vec[1], edge_vec[0]])
            denom = np.dot(edge_normal, ray_dir)
            if np.abs(denom) < 1e-6:
                continue  # Parallel, no intersection
            t = np.dot(edge_normal, p1 - ray_origin) / denom
            if t < 0:
                continue  # Intersection behind the ray origin
            intersection_point = ray_origin + t * ray_dir
            if np.dot(edge_vec, intersection_point - p1) >= 0 and np.dot(-edge_vec, intersection_point - p2) >= 0:
                depth = min(depth, t)
        return depth

def random_objs(num_objs):
    objs = []
    for _ in range(num_objs):
        if np.random.rand() < 0.5:
            width, height = np.random.uniform(0.1, 1.2, size=2)
            pos = np.random.uniform(-3, 3, size=2)
            rot = np.random.uniform(0, 360)
            objs.append(polygon2d.rectangle(width, height, pos, rot))
        else:
            radius = np.random.uniform(0.1, 0.6)
            pos = np.random.uniform(-3, 3, size=2)
            objs.append(polygon2d.circle(radius, pos))
    return objs

def boundary_rectangles(arena_size=4, boundary_size=0.5):
    out = []
    # Left boundary
    out.append(polygon2d.rectangle(boundary_size, arena_size * 2, pos=np.array([-arena_size - boundary_size / 2, 0]), rot=0))
    # Right boundary
    out.append(polygon2d.rectangle(boundary_size, arena_size * 2, pos=np.array([arena_size + boundary_size / 2, 0]), rot=0))
    # Top boundary
    out.append(polygon2d.rectangle(arena_size * 2, boundary_size, pos=np.array([0, arena_size + boundary_size / 2]), rot=0))
    # Bottom boundary
    out.append(polygon2d.rectangle(arena_size * 2, boundary_size, pos=np.array([0, -arena_size - boundary_size / 2]), rot=0))
    return out



class camera:
    def __init__(self, pos=np.array([0.0, 0.0]), rot=0.0, fov=90.0):
        self.pos = pos
        self.rot = rot
        self.fov = fov
        self.recorded_points = []

    def depth_view(self, objs, num_rays=10):
        angles = np.linspace(-self.fov/2, self.fov/2, num_rays)
        depths = []
        dirs = []
        for angle in angles:
            ray_angle = np.radians(self.rot + angle)
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            min_depth = float('inf')
            for obj in objs:
                depth = obj.closest_intersection(self.pos, ray_dir)
                depth*=np.random.normal(1,0.05)
                if depth < min_depth:
                    min_depth = depth
            depths.append(min_depth)
            dirs.append(ray_dir)
        return angles, depths, dirs

    def record_depths(self, objs, num_rays=10):
        angles, depths, dirs = self.depth_view(objs, num_rays)
        ends = []
        for angle, depth, dir in zip(angles, depths, dirs):
            end_point = self.pos + dir * depth
            ends.append(end_point)
        self.recorded_points.extend(ends)

    def plot(self):
        plt.plot(self.pos[0], self.pos[1], 'ro')
        theta = np.radians(self.rot)
        forward = np.array([np.cos(theta), np.sin(theta)]) * 1
        right_fov_lim = np.array([np.cos(theta + np.radians(self.fov/2)), np.sin(theta + np.radians(self.fov/2))]) * 1
        left_fov_lim = np.array([np.cos(theta - np.radians(self.fov/2)), np.sin(theta - np.radians(self.fov/2))]) * 1
        plt.plot([self.pos[0], self.pos[0] + forward[0]], [self.pos[1], self.pos[1] + forward[1]], 'r-')
        plt.plot([self.pos[0], self.pos[0] + right_fov_lim[0]], [self.pos[1], self.pos[1] + right_fov_lim[1]], 'r--')
        plt.plot([self.pos[0], self.pos[0] + left_fov_lim[0]], [self.pos[1], self.pos[1] + left_fov_lim[1]], 'r--')

    def plot_view(self, objs, num_rays=10):
        angles, depths, dirs = self.depth_view(objs, num_rays)
        ends = []
        for angle, depth, dir in zip(angles, depths, dirs):
            end_point = self.pos + dir * depth
            ends.append(end_point)
        ends = np.array(ends)
        plt.plot(ends[:, 0], ends[:, 1], 'r-o')

    def plot_memory(self):
        if self.recorded_points:
            points = np.array(self.recorded_points)
            plt.scatter(points[:, 0], points[:, 1])



def main() -> None:
    objs = random_objs(7)
    objs.extend(boundary_rectangles())
    for obj in objs:
        obj.plot()
    cam = camera(pos=np.array([0.0, 0.0]), rot=45.0)
    cam.plot()
    cam.plot_view(objs, num_rays=10)
    plt.show()
    for obj in objs:
        obj.plot()
    for _ in range(50):
        move_delta = np.random.uniform(-0.5, 0.5, size=2)
        rot_delta = np.random.uniform(-10, 30)
        cam.pos += move_delta
        cam.rot += rot_delta
        cam.record_depths(objs)
        cam.plot()
    plt.show()
    cam.plot_memory()
    plt.show()
    print(len(cam.recorded_points))

if __name__ == '__main__':
    main()