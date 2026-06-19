// hysteresis.cpp — Hysteresis Edge Tracing (Stage 6 of Canny)
//
// Resolves the STRONG / WEAK / NONE classification produced by
// doubleThreshold() into a final binary edge map.
//
// Rule: a STRONG pixel is always kept. A WEAK pixel is kept only if it is
// 8-connected (directly or transitively, through a chain of other WEAK
// pixels) to at least one STRONG pixel. Isolated WEAK pixels — and WEAK
// pixels only connected to other WEAK pixels with no STRONG anchor — are
// suppressed.
//
// Implementation: iterative flood-fill (BFS) seeded from every STRONG
// pixel, using an explicit stack/queue on the heap. Recursion is avoided
// since a naive recursive flood-fill could blow the stack on large images
// with long edge chains.

#include "hysteresis.h"
#include <cstring>
#include <cstdlib>
#include <vector>

void hysteresisEdgeTracing(
    const uint8_t* classified,
    uint8_t*       output,
    int            width,
    int            height)
{
    const int N = width * height;

    // visited[i] == true  → pixel has already been confirmed as a final edge
    std::vector<uint8_t> visited(N, 0);

    // Explicit work stack of pixel indices to flood from.
    std::vector<int> stack;
    stack.reserve(N / 4 + 16);

    // Seed the stack with every STRONG pixel and mark it visited immediately.
    for (int i = 0; i < N; i++)
    {
        if (classified[i] == EDGE_STRONG)
        {
            visited[i] = 1;
            stack.push_back(i);
        }
    }

    // 8-connected neighbour offsets (dx, dy)
    static const int dx[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
    static const int dy[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };

    while (!stack.empty())
    {
        const int idx = stack.back();
        stack.pop_back();

        const int x = idx % width;
        const int y = idx / width;

        for (int k = 0; k < 8; k++)
        {
            const int nx = x + dx[k];
            const int ny = y + dy[k];

            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            const int nidx = ny * width + nx;

            if (visited[nidx])
                continue;

            // Only WEAK (or STRONG, already handled) pixels propagate the
            // connection. NONE pixels block the flood.
            if (classified[nidx] == EDGE_WEAK || classified[nidx] == EDGE_STRONG)
            {
                visited[nidx] = 1;
                stack.push_back(nidx);
            }
        }
    }

    // Final output: visited pixels become confirmed edges (255), everything
    // else is suppressed to 0. Safe even if output == classified, since we
    // only read `classified` above and write `output` here, in that order
    // per index — but to be fully safe for in-place use with the visited
    // buffer already computed, just write fresh values now.
    for (int i = 0; i < N; i++)
        output[i] = visited[i] ? EDGE_STRONG : EDGE_NONE;
}

uint8_t* cannyFinalEdges(
    const uint8_t* nms_output,
    int            width,
    int            height,
    uint8_t        low,
    uint8_t        high)
{
    const size_t N = (size_t)width * height;

    uint8_t* classified = (uint8_t*)malloc(N);
    if (!classified)
        return nullptr;

    doubleThreshold(nms_output, classified, width, height, low, high);

    uint8_t* final_edges = (uint8_t*)malloc(N);
    if (!final_edges)
    {
        free(classified);
        return nullptr;
    }

    hysteresisEdgeTracing(classified, final_edges, width, height);

    free(classified);
    return final_edges;
}
