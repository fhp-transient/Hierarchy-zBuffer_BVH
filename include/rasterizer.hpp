//
// Created by goksu on 4/6/19.
//

#pragma once

#include <Eigen/Eigen>
#include <optional>
#include <algorithm>
#include "global.hpp"
#include "Shader.hpp"
#include "Triangle.hpp"

using namespace Eigen;

namespace rst
{
    enum class Buffers
    {
        Color = 1,
        Depth = 2
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    enum class Primitive
    {
        Line,
        Triangle
    };

    /*
     * For the curious : The draw function takes two buffer id's as its arguments. These two structs
     * make sure that if you mix up with their orders, the compiler won't compile it.
     * Aka : Type safety
     * */
    struct pos_buf_id
    {
        int pos_id = 0;
    };

    struct ind_buf_id
    {
        int ind_id = 0;
    };

    struct col_buf_id
    {
        int col_id = 0;
    };

    struct Node
    {
        Node* father;
        std::vector<Node*> children;
        int level;
        float depth;
        int l, r, d, u;
        bool visited = false; // 增加访问标记

        Node(Node* father, int level, float depth, int l, int r, int d, int u) : father(father), level(level),
            depth(depth), l(l), r(r), d(d), u(u)
        {
        };
    };

    struct BVHNode {
        BVHNode* left = nullptr;  // 左子节点
        BVHNode* right = nullptr; // 右子节点
        std::vector<Triangle*> triangles; // 直接保存三角形的指针
        Eigen::Vector3f bbox_min;         // 包围盒最小点
        Eigen::Vector3f bbox_max;         // 包围盒最大点

        BVHNode() = default;

        BVHNode(const Eigen::Vector3f& min, const Eigen::Vector3f& max)
            : bbox_min(min), bbox_max(max) {}
    };

    struct Edge
    {
        float x_up; // 边上端点的x坐标
        float y_up; // 边上端点的y坐标
        float x_down; // 边下端点的x坐标
        float y_down; // 边下端点的y坐标
        float dx; // -1 / k
        float z_up; // 边上端点的z坐标
        float z_down; // 边下端点的z坐标
    };

    class rasterizer
    {
    public:
        rasterizer(int w, int h);
        pos_buf_id load_positions(const std::vector<Eigen::Vector3f>& positions);
        ind_buf_id load_indices(const std::vector<Eigen::Vector3i>& indices);
        col_buf_id load_colors(const std::vector<Eigen::Vector3f>& colors);
        col_buf_id load_normals(const std::vector<Eigen::Vector3f>& normals);

        void set_model(const Eigen::Matrix4f& m);
        void set_view(const Eigen::Matrix4f& v);
        void set_projection(const Eigen::Matrix4f& p);

        void set_texture(Texture tex) { texture = tex; }

        void set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader);
        void set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader);

        void set_pixel(const Vector2i& point, const Eigen::Vector3f& color);

        void clear(Buffers buff);

        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type);
        void draw(std::vector<Triangle*>& TriangleList, int type);

        void set_color(float x, float y, const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos, const std::array<int, 3>& indices);

        std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

        void resetQuad();

    private:
        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);

        void rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos);

        std::unique_ptr<rst::Edge> buildEdge(Vector4f& v1, Vector4f& v2);

        void rasterize_triangle_scanlineZBuffer(Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos);

        void rasterize_triangle_BaseHierarchicalZBuffer(Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos);

        void BaseHierarchicalZBuffer(Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos);

        Node* buildQuadTree(Node* father, int level, int l, int r, int d, int u);

        BVHNode* buildBVH(std::vector<Triangle*>& triangles, int depth);

        void resetQuadTree(Node* v);

        void updateNode(Node* node);

        bool isBVHNodeVisible(BVHNode* node, Node* quadNode);

        void traverseBVHAndQuad(BVHNode* bvhNode, Node* quadNode);

        void traverseQuad(Node* quadNode, BVHNode* bvhRoot);

        void deleteBVH(BVHNode* node);

        // VERTEX SHADER -> MVP -> Clipping -> /.W -> VIEWPORT -> DRAWLINE/DRAWTRI -> FRAGSHADER

    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        int normal_id = -1;

        std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf;
        std::map<int, std::vector<Eigen::Vector3f>> col_buf;
        std::map<int, std::vector<Eigen::Vector3f>> nor_buf;

        std::optional<Texture> texture;

        std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;
        std::function<Eigen::Vector3f(vertex_shader_payload)> vertex_shader;

        std::vector<Eigen::Vector3f> frame_buf;
        std::vector<float> depth_buf;
        std::vector<Node*> zbufferToNode;
        int get_index(int x, int y);

        int width, height;

        int next_id = 0;
        int get_next_id() { return next_id++; }

        std::vector<Node*> nodes;

        Node* root;

        BVHNode* bvhRoot;

        std::queue<Node*> updateQueue;

        std::vector<Triangle *> m_TriangleList;

        std::vector<std::array<Eigen::Vector3f, 3> > m_viewspace_pos;
    };
}
