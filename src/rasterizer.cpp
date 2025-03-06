//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f>& positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i>& indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f>& cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


void rst::rasterizer::resetQuad()
{
    resetQuadTree(root);
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1)
    {
        if (dx >= 0)
        {
            x = x1;
            y = y1;
            xe = x2;
        }
        else
        {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++)
        {
            x = x + 1;
            if (px < 0)
            {
                px = px + 2 * dy1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    y = y + 1;
                }
                else
                {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
    else
    {
        if (dy >= 0)
        {
            x = x1;
            y = y1;
            ye = y2;
        }
        else
        {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++)
        {
            y = y + 1;
            if (py <= 0)
            {
                py = py + 2 * dx1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    x = x + 1;
                }
                else
                {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector4f* _v)
{
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0, f1, f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x, y, 1.);
    if ((p.dot(f0) * f0.dot(v[2]) > 0) && (p.dot(f1) * f1.dot(v[0]) > 0) && (p.dot(f2) * f2.dot(v[1]) > 0))
        return true; // 判断点是否在同侧
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[
        0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[
        1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[
        2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::draw(std::vector<Triangle*>& TriangleList, int type)
{
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto& t : TriangleList)
    {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{
            (view * model * t->v[0]),
            (view * model * t->v[1]),
            (view * model * t->v[2])
        };

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v)
        {
            return v.template head<3>();
        });

        Eigen::Vector4f v[] = {
            mvp * t->v[0],
            mvp * t->v[1],
            mvp * t->v[2]
        };
        //Homogeneous division
        for (auto& vec : v)
        {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
            inv_trans * to_vec4(t->normal[0], 0.0f),
            inv_trans * to_vec4(t->normal[1], 0.0f),
            inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148.0, 121.0, 92.0);
        newtri.setColor(1, 148.0, 121.0, 92.0);
        newtri.setColor(2, 148.0, 121.0, 92.0);

        m_TriangleList.push_back(new Triangle(newtri));
        m_viewspace_pos.push_back(viewspace_pos);
        // Also pass view space vertice position
        // rasterize_triangle(newtri, viewspace_pos);
        // rasterize_triangle_scanlineZBuffer(newtri, viewspace_pos);
        // BaseHierarchicalZBuffer(newtri, viewspace_pos);
    }

    for (int i = 0; i < m_TriangleList.size(); i++)
    {
        m_TriangleList[i]->id = i;
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;

    switch (type)
    {
    case 0:
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m_TriangleList.size(); i++)
        {
            // rasterize_triangle(newtri, viewspace_pos);
            rasterize_triangle_scanlineZBuffer(*m_TriangleList[i], m_viewspace_pos[i]);
            // BaseHierarchicalZBuffer(*m_TriangleList[i], m_viewspace_pos[i]);
        }
        end = std::chrono::high_resolution_clock::now();

        std::cout << "scanlineZBuffer Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).
            count() << "ms" << std::endl;

        break;
    case 1:
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m_TriangleList.size(); i++)
        {
            // rasterize_triangle(newtri, viewspace_pos);
            // rasterize_triangle_scanlineZBuffer(*newTriangleList[i], m_viewspace_pos[i]);
            BaseHierarchicalZBuffer(*m_TriangleList[i], m_viewspace_pos[i]);
        }
        end = std::chrono::high_resolution_clock::now();

        std::cout << "BaseHierarchicalZBuffer Time: " << std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        break;
    case 2:
        bvhRoot = buildBVH(m_TriangleList, 0);
        start = std::chrono::high_resolution_clock::now();
        traverseBVHAndQuad(bvhRoot, root);
        end = std::chrono::high_resolution_clock::now();

        std::cout << "BvhHierarchicalZBuffer Time: " << std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        deleteBVH(bvhRoot);
        break;
    default:
        std::cout << "Wrong type, type from 0 to 2!" << std::endl;
        return;
    }

    for (int i = 0; i < m_TriangleList.size(); i++)
    {
        delete(m_TriangleList[i]);
    }
    m_TriangleList.clear();
    m_viewspace_pos.clear();
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
                                   const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f& vert1,
                                   const Eigen::Vector2f& vert2, const Eigen::Vector2f& vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

void rst::rasterizer::set_color(float x, float y, const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos,
                                const std::array<int, 3>& indices)
{
    auto v = t.v;
    Vector4f vsorted[] = {v[indices[0]], v[indices[1]], v[indices[2]]};
    auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, vsorted);
    float w_reciprocal = 1.0f / (alpha / v[indices[0]].w() + beta / v[indices[1]].w() + gamma / v[indices[2]].w());

    auto interpolated_color = interpolate(alpha / v[indices[0]].w(), beta / v[indices[1]].w(),
                                          gamma / v[indices[2]].w(),
                                          t.color[indices[0]], t.color[indices[1]], t.color[indices[2]],
                                          1 / w_reciprocal);
    // auto interpolated_normal
    auto interpolated_normal = interpolate(alpha / v[indices[0]].w(), beta / v[indices[1]].w(),
                                           gamma / v[indices[2]].w(),
                                           t.normal[indices[0]], t.normal[indices[1]], t.normal[indices[2]],
                                           1 / w_reciprocal);
    // auto interpolated_texcoords
    auto interpolated_texcoords = interpolate(alpha / v[indices[0]].w(), beta / v[indices[1]].w(),
                                              gamma / v[indices[2]].w(),
                                              t.tex_coords[indices[0]], t.tex_coords[indices[1]],
                                              t.tex_coords[indices[2]],
                                              1 / w_reciprocal);
    // auto interpolated_shadingcoords
    auto interpolated_shadingcoords = interpolate(alpha / v[indices[0]].w(), beta / v[indices[1]].w(),
                                                  gamma / v[indices[2]].w(),
                                                  view_pos[indices[0]], view_pos[indices[1]], view_pos[indices[2]],
                                                  1 / w_reciprocal);
    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(),
                                    interpolated_texcoords, texture ? &*texture : nullptr);
    payload.view_pos = interpolated_shadingcoords;
    auto pixel_color = fragment_shader(payload);
    set_pixel(Eigen::Vector2i(x, y), pixel_color);
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos)
{
    // TODO: From your HW3, get the triangle rasterization code.
    auto v = t.v;
    int xmin = INT_MAX, xmax = -INT_MAX, ymax = -INT_MAX, ymin = INT_MAX;
    for (int i = 0; i < 3; i++)
    {
        xmin = std::min(xmin, (int)v[i].x());
        xmax = ceil(std::max(xmax, (int)v[i].x()));
        ymin = std::min(ymin, (int)v[i].y());
        ymax = ceil(std::max(ymax, (int)v[i].y()));
    }
    xmax = std::min(xmax, width - 1);
    ymax = std::min(ymax, height - 1);
    // TODO: Inside your rasterization loop:
    //    * v[i].w() is the vertex view space depth value z.
    //    * Z is interpolated view space depth for the current pixel
    //    * zp is depth between zNear and zFar, used for z-buffer
    for (int x = xmin; x <= xmax; x++)
    {
        for (int y = ymin; y <= ymax; y++)
        {
            if (insideTriangle(x + 0.5, y + 0.5, t.v))
            {
                auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, t.v);
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated =
                    alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                int ind = get_index(x, y);
                if (z_interpolated > depth_buf[ind])
                {
                    depth_buf[ind] = z_interpolated;
                    // TODO: Interpolate the attributes:
                    // auto interpolated_color
                    auto interpolated_color = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(),
                                                          t.color[0], t.color[1], t.color[2], 1 / w_reciprocal);
                    // auto interpolated_normal
                    auto interpolated_normal = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(),
                                                           t.normal[0], t.normal[1], t.normal[2], 1 / w_reciprocal);
                    // auto interpolated_texcoords
                    auto interpolated_texcoords = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(),
                                                              t.tex_coords[0], t.tex_coords[1], t.tex_coords[2],
                                                              1 / w_reciprocal);
                    // auto interpolated_shadingcoords
                    auto interpolated_shadingcoords = interpolate(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w(),
                                                                  view_pos[0], view_pos[1], view_pos[2],
                                                                  1 / w_reciprocal);
                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(),
                                                    interpolated_texcoords, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;
                    auto pixel_color = fragment_shader(payload);
                    set_pixel(Eigen::Vector2i(x, y), pixel_color);
                }
            }
        }
    }

    // Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
    // Use: payload.view_pos = interpolated_shadingcoords;
    // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
    // Use: auto pixel_color = fragment_shader(payload);
}

std::unique_ptr<rst::Edge> rst::rasterizer::buildEdge(Vector4f& v1, Vector4f& v2)
{
    auto edge = std::make_unique<Edge>();
    edge->x_up = v1.x();
    edge->y_up = v1.y();
    edge->x_down = v2.x();
    edge->y_down = v2.y();
    if (abs(v1.y() - v2.y()) > .00001)
        edge->dx = (v2.x() - v1.x()) / ((v1.y() - v2.y()));
    else edge->dx = 0;
    edge->z_up = v1.z();
    edge->z_down = v2.z();
    return edge;
}


void rst::rasterizer::rasterize_triangle_scanlineZBuffer(Triangle& t,
                                                         const std::array<Eigen::Vector3f, 3>& view_pos)
{
    float epsilon = 1e-5;

    auto v = t.v; // 三角形顶点
    // 按顶点 y 坐标排序，确保 v0.y >= v[indices[0]].y >= v[indices[1]].y
    std::array<int, 3> indices = {0, 1, 2};
    std::sort(indices.begin(), indices.end(), [&v](int a, int b)
    {
        return v[a].y() > v[b].y(); // 按 y 坐标降序排序
    });

    // 建立边
    auto e1 = buildEdge(v[indices[0]], v[indices[1]]);
    auto e2 = buildEdge(v[indices[0]], v[indices[2]]);
    auto e3 = buildEdge(v[indices[1]], v[indices[2]]);

    // 计算三角形方程..
    float a = (v[indices[1]].y() - v[indices[0]].y()) * (v[indices[2]].z() - v[indices[0]].z()) - (v[indices[1]].z() - v
        [indices[0]].z()) * (v[indices[2]].y() - v[indices[0]].y());
    float b = (v[indices[1]].z() - v[indices[0]].z()) * (v[indices[2]].x() - v[indices[0]].x()) - (v[indices[1]].x() - v
        [indices[0]].x()) * (v[indices[2]].z() - v[indices[0]].z());
    float c = (v[indices[1]].x() - v[indices[0]].x()) * (v[indices[2]].y() - v[indices[0]].y()) - (v[indices[1]].y() - v
        [indices[0]].y()) * (v[indices[2]].x() - v[indices[0]].x());
    float d = -(a * v[indices[0]].x() + b * v[indices[0]].y() + c * v[indices[0]].z());

    bool flag = false;
    if (std::abs(c) < 1e-3)
    {
        flag = true;
    }

    // 计算三角形分割点
    int yu = std::floor(v[indices[0]].y());
    int ym = std::floor(v[indices[1]].y());
    int yd = std::floor(v[indices[2]].y());

    // e1在左，e2在右，画上半三角
    if (e1->x_down > e2->x_down)
        std::swap(e1, e2);
    float xl = std::max(0, int(e1->x_up));
    float xr = std::min(WINDOW_WIDTH - 1, int(e2->x_up));
    float zl = e1->z_up;
    float dzx = 0, dzy = 0;
    if (!flag)
    {
        dzx = -a / c;
        dzy = b / c;
    }

    for (int y = yu + 1; y > ym; y--)
    {
        float z = zl;
        int intxl = std::floor(xl);

        if (intxl < 10) std::cout << "xl: " << xl << " xr: " << xr << "y: " << y << "z: " << z << std::endl;

        int intxr = std::floor(xr);

        for (int x = intxl; x <= intxr; x++)
        {
            int ind = get_index(x, y);
            // if (z > Szbuffer[x][y])
            if (z > depth_buf[ind])
            {
                // 更新zbuffer的深度
                depth_buf[ind] = z;
                // auto v = t.v;
                set_color(x, y, t, view_pos, indices);
            }
            z += dzx;
        }
        // 活化边推进到下一条扫描线
        xl += e1->dx;
        xr += e2->dx;
        zl += dzx * e1->dx + dzy;
    }

    // e2在左，e3在右，画下半三角
    if (e1->y_down < e2->y_down)
        std::swap(e1, e2);
    if (e2->x_up > e3->x_up)
        std::swap(e2, e3);

    xl = e2->x_up + e2->dx * (e2->y_up - ym);
    xr = e3->x_up + e3->dx * (e3->y_up - ym);
    zl = e2->z_up + e2->dx * (e2->y_up - ym) * dzx + dzy * (e2->y_up - ym);

    for (int y = ym; y >= yd; y--)
    {
        float z = zl;
        int intxl = std::floor(xl);
        int intxr = std::floor(xr);

        for (int x = intxl; x <= intxr; x++)
        {
            int ind = get_index(x, y);
            // if (z > Szbuffer[x][y])
            if (z > depth_buf[ind])
            {
                // 更新zbuffer的深度
                depth_buf[ind] = z;
                // auto v = t.v;
                set_color(x, y, t, view_pos, indices);
            }
            z += dzx;
        }
        // 活化边推进到下一条扫描线
        xl += e2->dx;
        xr += e3->dx;
        zl += dzx * e2->dx + dzy;
    }
}

void rst::rasterizer::rasterize_triangle_BaseHierarchicalZBuffer(Triangle& t,
                                                                 const std::array<Eigen::Vector3f, 3>& view_pos)
{
    auto v = t.v; // 三角形顶点
    // 按顶点 y 坐标排序，确保 v0.y >= v[indices[0]].y >= v[indices[1]].y
    std::array<int, 3> indices = {0, 1, 2};
    std::sort(indices.begin(), indices.end(), [&v](int a, int b)
    {
        return v[a].y() > v[b].y(); // 按 y 坐标降序排序
    });

    Vector4f vsorted[] = {v[indices[0]], v[indices[1]], v[indices[2]]};

    // 建立边
    auto e1 = buildEdge(v[indices[0]], v[indices[1]]);
    auto e2 = buildEdge(v[indices[0]], v[indices[2]]);
    auto e3 = buildEdge(v[indices[1]], v[indices[2]]);

    // 计算三角形方程..
    float a = (v[indices[1]].y() - v[indices[0]].y()) * (v[indices[2]].z() - v[indices[0]].z()) - (v[indices[1]].z() - v
        [indices[0]].z()) * (v[indices[2]].y() - v[indices[0]].y());
    float b = (v[indices[1]].z() - v[indices[0]].z()) * (v[indices[2]].x() - v[indices[0]].x()) - (v[indices[1]].x() - v
        [indices[0]].x()) * (v[indices[2]].z() - v[indices[0]].z());
    float c = (v[indices[1]].x() - v[indices[0]].x()) * (v[indices[2]].y() - v[indices[0]].y()) - (v[indices[1]].y() - v
        [indices[0]].y()) * (v[indices[2]].x() - v[indices[0]].x());
    float d = -(a * v[indices[0]].x() + b * v[indices[0]].y() + c * v[indices[0]].z());

    bool flag = false;
    if (std::abs(c) < 1e-6)
    {
        flag = true;
    }

    // 计算三角形分割点
    int yu = int(v[indices[0]].y());
    int ym = int(v[indices[1]].y());
    int yd = int(v[indices[2]].y());

    // e1在左，e2在右，画上半三角
    if (e1->x_down > e2->x_down)
        std::swap(e1, e2);
    float xl = std::max(0, int(e1->x_up));
    float xr = std::min(WINDOW_WIDTH - 1, int(e2->x_up));
    float zl = e1->z_up;
    float dzx = 0, dzy = 0;
    if (!flag)
    {
        dzx = -a / c;
        dzy = b / c;
    }

    for (int y = yu; y > ym; y--)
    {
        float z = zl;
        int intxl = int(xl);
        int intxr = int(xr);

        for (int x = intxl; x <= intxr; x++)
        {
            int ind = get_index(x, y);
            // if (z > Szbuffer[x][y])
            if (z > depth_buf[ind])
            {
                if (abs(zbufferToNode[ind]->depth - depth_buf[ind]) < .000001)
                {
                    updateQueue.push(zbufferToNode[ind]);
                }

                // 更新zbuffer的深度
                depth_buf[ind] = z;
                // auto v = t.v;
                set_color(x, y, t, view_pos, indices);
            }
            z += dzx;
        }
        // 活化边推进到下一条扫描线
        xl += e1->dx;
        xr += e2->dx;
        zl += dzx * e1->dx + dzy;
    }

    // e2在左，e3在右，画下半三角
    if (e1->y_down < e2->y_down)
        std::swap(e1, e2);
    if (e2->x_up > e3->x_up)
        std::swap(e2, e3);

    xl = e2->x_up + e2->dx * (e2->y_up - ym);
    xr = e3->x_up + e3->dx * (e3->y_up - ym);
    zl = e2->z_up + e2->dx * (e2->y_up - ym) * dzx + dzy * (e2->y_up - ym);

    for (int y = ym; y > yd; y--)
    {
        float z = zl;
        int intxl = int(xl);
        int intxr = int(xr);

        for (int x = intxl; x <= intxr; x++)
        {
            int ind = get_index(x, y);
            // if (z > Szbuffer[x][y])
            if (z > depth_buf[ind])
            {
                if (abs(zbufferToNode[ind]->depth - depth_buf[ind]) < .000001)
                {
                    updateQueue.push(zbufferToNode[ind]);
                }

                // 更新zbuffer的深度
                depth_buf[ind] = z;
                // auto v = t.v;
                set_color(x, y, t, view_pos, indices);
            }
            z += dzx;
        }
        // 活化边推进到下一条扫描线
        xl += e2->dx;
        xr += e3->dx;
        zl += dzx * e2->dx + dzy;
    }

    Node* prevNode = nullptr; // 用于保存前一个节点
    int queueSize = updateQueue.size();

    for (int i = 0; i < queueSize; i++)
    {
        Node* currentNode = updateQueue.front(); // 获取队列头部的元素
        updateQueue.pop(); // 弹出队列头部的元素

        if (prevNode == nullptr || prevNode != currentNode)
        {
            updateNode(currentNode); // 处理当前节点
        }

        prevNode = currentNode; // 更新 prevNode
    }
}

void rst::rasterizer::BaseHierarchicalZBuffer(Triangle& t,
                                              const std::array<Eigen::Vector3f, 3>& view_pos)
{
    // resetQuadTree(root);
    auto v = t.v;
    // Vertex* v1 = &vertices[faces[i].v1];
    // Vertex* v2 = &vertices[faces[i].v2];[1].
    // Vertex* v3 = &vertices[faces[i].v3];[0]
    Node* v1n = zbufferToNode[get_index(int(v[0].x()), int(v[0].y()))];
    Node* v2n = zbufferToNode[get_index(int(v[1].x()), int(v[1].y()))];
    Node* v3n = zbufferToNode[get_index(int(v[2].x()), int(v[2].y()))];

    // std::cout << "v[0]:" << v[0].x() << " " << v[0].y() << std::endl;
    // std::cout << "v[1]:" << v[1].x() << " " << v[1].y() << std::endl;
    // std::cout << "v[2]:" << v[2].x() << " " << v[2].y() << std::endl;

    // 自底向上找最小共同祖先
    while (v1n->level > v2n->level)
        v1n = v1n->father;
    while (v2n->level > v1n->level)
        v2n = v2n->father;
    while (v1n->level > v3n->level)
        v1n = v1n->father;
    while (v3n->level > v1n->level)
        v3n = v3n->father;
    while (v2n->level > v3n->level)
        v2n = v2n->father;
    while (v3n->level > v2n->level)
        v3n = v3n->father;
    while (!(v1n == v2n && v2n == v3n))
    {
        v1n = v1n->father;
        v2n = v2n->father;
        v3n = v3n->father;
    }

    // 若三角形最大z值大于祖先结点，则可光栅化之
    float depth = v1n->depth;
    if (v[0].z() > depth || v[1].z() > depth || v[2].z() > depth)
        rasterize_triangle_BaseHierarchicalZBuffer(t, view_pos);
}

rst::Node* rst::rasterizer::buildQuadTree(Node* father, int level, int l, int r, int d, int u)
{
    // // // 格子面积为0，不建立
    // if (l == r || d == u)
    //     return nullptr;
    // // 格子恰好为一个像素，进入leaves成为叶子结点
    // if (l == r - 1 && d == u - 1)
    // {
    //     int idx = get_index(l, d);
    //     zbufferToNode[idx] = new Node(father, level, -1e5, l, r, d, u);
    //     //printf("built %d %d\n", l, d);
    //     return zbufferToNode[idx];
    // }
    // // 格子非叶子且有面积，继续拆分
    // Node* quadNode = new Node(father, level, -1e5, l, r, d, u);
    // int lrmid = (l + r) / 2;
    // int dumid = (d + u) / 2;
    //
    // quadNode->children.push_back(buildQuadTree(quadNode, level + 1, l, lrmid, d, dumid));
    // quadNode->children.push_back(buildQuadTree(quadNode, level + 1, l, lrmid, dumid, u));
    // quadNode->children.push_back(buildQuadTree(quadNode, level + 1, lrmid, r, d, dumid));
    // quadNode->children.push_back(buildQuadTree(quadNode, level + 1, lrmid, r, dumid, u));
    // return quadNode;
    // 格子面积为0，不建立
    if (l >= r || d >= u)
        return nullptr;
    // 格子有面积，允许建立结点
    Node* node = new Node(father, level, -1e5, l, r, d, u);
    nodes.push_back(node);

    // 四叉树深度未逾越限制，且本结点不是单个像素点，则继续分割
    if (level < MAX_QUAD_LEVEL && !(l == r - 1 && d == u - 1))
    {
        int lrmid = l + (r - l) / 2;
        int dumid = d + (u - d) / 2;
        node->children.push_back(buildQuadTree(node, level + 1, l, lrmid, d, dumid));
        node->children.push_back(buildQuadTree(node, level + 1, l, lrmid, dumid, u));
        node->children.push_back(buildQuadTree(node, level + 1, lrmid, r, d, dumid));
        node->children.push_back(buildQuadTree(node, level + 1, lrmid, r, dumid, u));
    }
    else // leaf node
    {
        for (int x = l; x < r; x++)
        {
            for (int y = d; y < u; y++)
            {
                int idx = get_index(x, y);
                zbufferToNode[idx] = node;
            }
        }
    }
    return node;
}

rst::BVHNode* rst::rasterizer::buildBVH(std::vector<Triangle*>& triangles, int depth)
{
    if (triangles.empty()) return nullptr;

    // 计算包围盒
    Eigen::Vector3f bbox_min = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f bbox_max = Eigen::Vector3f::Constant(-FLT_MAX);
    for (const auto& triangle : triangles)
    {
        for (int i = 0; i < 3; i++)
        {
            bbox_min = bbox_min.cwiseMin(triangle->v[i].head<3>());
            bbox_max = bbox_max.cwiseMax(triangle->v[i].head<3>());
        }
    }

    // 创建当前 BVH 节点
    BVHNode* node = new BVHNode(bbox_min, bbox_max);

    // 如果是叶子节点，直接保存三角形并返回
    if (triangles.size() <= MAX_TRIANGLES_PER_NODE)
    {
        node->triangles = triangles;
        return node;
    }

    // 根据深度选择分割轴
    int axis = depth % 3; // 0 = x, 1 = y, 2 = z
    std::sort(triangles.begin(), triangles.end(), [&](Triangle* a, Triangle* b)
    {
        float centerA = (a->v[0][axis] + a->v[1][axis] + a->v[2][axis]) / 3.0f;
        float centerB = (b->v[0][axis] + b->v[1][axis] + b->v[2][axis]) / 3.0f;
        return centerA < centerB;
    });

    // 分割三角形集合
    size_t mid = triangles.size() / 2;
    std::vector<Triangle*> leftTriangles(triangles.begin(), triangles.begin() + mid);
    std::vector<Triangle*> rightTriangles(triangles.begin() + mid, triangles.end());

    // 递归构建左右子节点
    node->left = buildBVH(leftTriangles, depth + 1);
    node->right = buildBVH(rightTriangles, depth + 1);

    return node;
}

void rst::rasterizer::resetQuadTree(Node* v)
{
    if (!v) return;
    v->depth = -1e5;
    v->visited = false;
    for (Node* child : v->children)
    {
        if (child)
            resetQuadTree(child);
    }
}

void rst::rasterizer::updateNode(Node* node)
{
    float minDepth = 1e5;

    // 叶子结点
    if (node->children.empty())
    {
        for (int x = node->l; x < node->r; x++)
        {
            for (int y = node->d; y < node->u; y++)
            {
                int idx = get_index(x, y);
                if (depth_buf[idx] < minDepth)
                    minDepth = depth_buf[idx];
            }
        }
    }
    // 非叶子结点
    else
    {
        for (Node* child : node->children)
        {
            if (child && child->depth < minDepth)
                minDepth = child->depth;
        }
    }

    if (abs(minDepth - node->depth) > 0.000001) // not equal
    {
        node->depth = minDepth;
        if (node->father)
            updateNode(node->father);
    }
}

bool rst::rasterizer::isBVHNodeVisible(BVHNode* bvhNode, Node* quadNode)
{
    // 屏幕空间的包围盒
    float screen_left = quadNode->l;
    float screen_right = quadNode->r;
    float screen_top = quadNode->u;
    float screen_bottom = quadNode->d;

    // 判断 BVH 节点的包围盒是否与屏幕范围相交
    if (bvhNode->bbox_max.x() < screen_left || bvhNode->bbox_min.x() > screen_right ||
        bvhNode->bbox_max.y() < screen_bottom || bvhNode->bbox_min.y() > screen_top)
    {
        return false;
    }

    // 判断深度遮挡
    if (quadNode->depth >= bvhNode->bbox_max.z())
    {
        return false;
    }

    return true;
}

void rst::rasterizer::traverseBVHAndQuad(BVHNode* bvhNode, Node* quadNode)
{
    if (!bvhNode || !isBVHNodeVisible(bvhNode, quadNode)) return;

    // 如果是叶子节点，直接处理其中的三角形
    if (!bvhNode->left && !bvhNode->right)
    {
        for (Triangle* triangle : bvhNode->triangles)
        {
            BaseHierarchicalZBuffer(*triangle, m_viewspace_pos[triangle->id]);
        }
        return;
    }

    // 如果 BVH 节点有子节点，递归处理左右子节点
    traverseBVHAndQuad(bvhNode->left, quadNode);
    traverseBVHAndQuad(bvhNode->right, quadNode);
}

void rst::rasterizer::traverseQuad(Node* quadNode, BVHNode* bvhRoot)
{
    if (!quadNode || quadNode->visited) return;
    quadNode->visited = true;

    // 遍历四叉树的子节点
    for (Node* child : quadNode->children)
    {
        if (child) traverseQuad(child, bvhRoot);
    }

    // 对当前四叉树节点调用 BVH 遍历
    traverseBVHAndQuad(bvhRoot, quadNode);
}

void rst::rasterizer::deleteBVH(BVHNode* node)
{
    if (!node) return;
    deleteBVH(node->left);
    deleteBVH(node->right);
    delete node;
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), -1e5);
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    zbufferToNode.resize(w * h);
    texture = std::nullopt;

    root = buildQuadTree(nullptr, 0, 0, WINDOW_WIDTH, 0, WINDOW_HEIGHT);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - y - 1) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y() - 1) * width + point.x();
    //    std::cout << point.x() << " " << point.y() << std::endl;
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}
