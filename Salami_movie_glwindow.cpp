#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

extern "C" {
    #include "libmeshb7.h"
}

#define PLANE_X 0
#define PLANE_Y 1
#define PLANE_Z 2

// Timer
double millitime() {
    struct timeval tp;
    if(gettimeofday(&tp, nullptr))
        return 0;
    return 1000.0 * tp.tv_sec + 0.001 * tp.tv_usec;
}

// Callback de redimensionnement
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Callback d'erreur
void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d): %s\n", error, description);
}

// Fonctions de compilation de shader
GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512] = {0};
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "Erreur compilation shader : %s\n", infoLog);
    }
    return shader;
}

GLuint createProgram(const char* vsSource, const char* fsSource) {
    GLuint vs = compileShader(vsSource, GL_VERTEX_SHADER);
    GLuint fs = compileShader(fsSource, GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success) {
        char infoLog[512] = {0};
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "Erreur linkage programme : %s\n", infoLog);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

GLuint compileComputeShader(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        char infoLog[512] = {0};
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "Erreur compilation compute shader : %s\n", infoLog);
    }
    return shader;
}

GLuint createComputeProgram(const char* csSource) {
    GLuint cs = compileComputeShader(csSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, cs);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success) {
        char infoLog[512] = {0};
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "Erreur linkage compute programme : %s\n", infoLog);
    }
    glDeleteShader(cs);
    return program;
}

// Structures côté CPU pour le maillage
struct Vertex {
    double x, y, z;
};

struct Tetrahedron {
    int v[4];
};

struct BoundingBox {
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

struct GridCell {
    int offset;
    int count;
};

// Calcul de la bounding box
BoundingBox calculate_bounding_box(int Nverts, double *xx, double *yy, double *zz) {
    BoundingBox bbox;
    if (Nverts < 1) {
        fprintf(stderr, "Aucun vertex pour calculer la bounding box.\n");
        exit(EXIT_FAILURE);
    }
    double *ptr_x = xx + 1, *ptr_y = yy + 1, *ptr_z = zz + 1;
    bbox.xmin = bbox.xmax = *ptr_x;
    bbox.ymin = bbox.ymax = *ptr_y;
    bbox.zmin = bbox.zmax = *ptr_z;
    for (int i = 1; i < Nverts; i++) {
        ptr_x++; ptr_y++; ptr_z++;
        if (*ptr_x < bbox.xmin) bbox.xmin = *ptr_x;
        if (*ptr_x > bbox.xmax) bbox.xmax = *ptr_x;
        if (*ptr_y < bbox.ymin) bbox.ymin = *ptr_y;
        if (*ptr_y > bbox.ymax) bbox.ymax = *ptr_y;
        if (*ptr_z < bbox.zmin) bbox.zmin = *ptr_z;
        if (*ptr_z > bbox.zmax) bbox.zmax = *ptr_z;
    }
    return bbox;
}

// Shaders pour le rendu
const char* quadVertexShaderSource = R"(
#version 430
layout(location = 0) in vec2 inPos;
out vec2 uv;
void main(){
    uv = inPos * 0.5 + 0.5;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
)";

const char* quadFragmentShaderSource = R"(
#version 430
in vec2 uv;
out vec4 fragColor;
uniform sampler2D computedTexture;
void main(){
    fragColor = texture(computedTexture, uv);
}
)";

// Compute shader (avec chargement du maillage, grille, etc.)
const char* computeShaderSource = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
uniform ivec2 imageSize;
uniform float planeVal;
uniform int planeAxis;
uniform float projMin1;
uniform float projMax1;
uniform float projMin2;
uniform float projMax2;
uniform float u_maxAbs;
uniform int nCells_u;
uniform int nCells_v;
struct FVertex { float x, y, z; };
struct FTetrahedron { ivec4 v; };
layout(std430, binding = 0) buffer VertexBuffer { FVertex vertices[]; };
layout(std430, binding = 1) buffer TetrahedronBuffer { FTetrahedron tets[]; };
layout(std430, binding = 2) buffer SolutionBuffer { float sol[]; };
struct GridCell { int offset; int count; };
layout(std430, binding = 4) buffer GridCellsBuffer { GridCell gridCells[]; };
layout(std430, binding = 5) buffer GridIndicesBuffer { int gridIndices[]; };
layout(rgba8, binding = 0) uniform writeonly image2D destImage;
float det3(vec3 a, vec3 b, vec3 c) { return dot(a, cross(b, c)); }
bool computeBarycentric(vec3 p, FTetrahedron tet, out vec4 coords) {
    vec3 p0 = vec3(vertices[tet.v.x].x, vertices[tet.v.x].y, vertices[tet.v.x].z);
    vec3 p1 = vec3(vertices[tet.v.y].x, vertices[tet.v.y].y, vertices[tet.v.y].z);
    vec3 p2 = vec3(vertices[tet.v.z].x, vertices[tet.v.z].y, vertices[tet.v.z].z);
    vec3 p3 = vec3(vertices[tet.v.w].x, vertices[tet.v.w].y, vertices[tet.v.w].z);
    vec3 v0 = p1 - p0, v1 = p2 - p0, v2 = p3 - p0;
    float detT = det3(v0, v1, v2);
    if (abs(detT) < 1e-6) return false;
    vec3 vp = p - p0;
    float u = det3(vp, v1, v2) / detT;
    float v = det3(v0, vp, v2) / detT;
    float w = det3(v0, v1, vp) / detT;
    float t_val = 1.0 - u - v - w;
    coords = vec4(t_val, u, v, w);
    float eps = 1e-6;
    return (u >= -eps && v >= -eps && w >= -eps && t_val >= -eps &&
            abs(u+v+w+t_val - 1.0) < eps);
}
vec3 mapToColor(float value, float maxAbs) {
    float normalized = clamp(value / maxAbs, -1.0, 1.0);
    vec3 col = vec3(0.0);
    if (normalized < 0.0) { col.r = 0.0; col.g = sqrt(abs(normalized)) * 0.5; col.b = sqrt(abs(normalized)); }
    else { col.r = normalized; col.g = 1.0 - normalized; col.b = 0.0; }
    return col;
}
void main(){
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;
    vec2 uv = vec2(pixelCoord) / vec2(imageSize);
    float coord1 = mix(projMin1, projMax1, uv.x);
    float coord2 = mix(projMin2, projMax2, uv.y);
    vec3 p;
    if (planeAxis == 0) p = vec3(planeVal, coord1, coord2);
    else if (planeAxis == 1) p = vec3(coord1, planeVal, coord2);
    else p = vec3(coord1, coord2, planeVal);
    float interpolated = 0.0;
    bool found = false;
    vec4 bary;
    int cell_u = int((coord1 - projMin1) / ((projMax1 - projMin1) / float(nCells_u)));
    int cell_v = int((coord2 - projMin2) / ((projMax2 - projMin2) / float(nCells_v)));
    cell_u = clamp(cell_u, 0, nCells_u - 1);
    cell_v = clamp(cell_v, 0, nCells_v - 1);
    int cellIndex = cell_u * nCells_v + cell_v;
    GridCell cell = gridCells[cellIndex];
    for (int i = 0; i < cell.count; i++){
        int tetIndex = gridIndices[cell.offset + i];
        if (computeBarycentric(p, tets[tetIndex], bary)) {
            float s0 = sol[tets[tetIndex].v.x];
            float s1 = sol[tets[tetIndex].v.y];
            float s2 = sol[tets[tetIndex].v.z];
            float s3 = sol[tets[tetIndex].v.w];
            interpolated = bary.x * s0 + bary.y * s1 + bary.z * s2 + bary.w * s3;
            found = true;
            break;
        }
    }
    vec3 color = found ? mapToColor(interpolated, u_maxAbs) : vec3(1.0, 0.0, 1.0);
    imageStore(destImage, pixelCoord, vec4(color, 1.0));
}
)";

int main() {
    const int width = 1024, height = 1024;

    // Initialisation de GLFW et création de la fenêtre
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()){
        fprintf(stderr, "Echec de l'initialisation de GLFW\n");
        return -1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // We delay the creation of the window until the first pictures 
    // are ready, to prevent a force quit/wait message from appearing
    //GLFWwindow* window = nullptr;

    GLFWwindow* window = glfwCreateWindow(width, height, "Mesh Compute Film", nullptr, nullptr);
    if (!window){
        fprintf(stderr, "Echec de la création de la fenêtre\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
        fprintf(stderr, "Echec de l'initialisation de GLAD\n");
        return -1;
    }
    glViewport(0, 0, width, height);


    // ----------------- Chargement du maillage et de la solution -----------------
    printf("Chargement du maillage...\n");
    int r;
    int64_t id_mesh, Nverts, Ntets;
    const char* meshFile = "smallo.mesh";
    int id_mesh_version, id_mesh_dim;
    id_mesh = GmfOpenMesh(meshFile, GmfRead, &id_mesh_version, &id_mesh_dim);
    if (!id_mesh) {
        fprintf(stderr, "Impossible d'ouvrir le fichier mesh %s\n", meshFile);
        return -1;
    }
    Nverts = GmfStatKwd(id_mesh, GmfVertices);
    Ntets  = GmfStatKwd(id_mesh, GmfTetrahedra);
    double* xx = (double*) malloc((Nverts+1)*sizeof(double));
    double* yy = (double*) malloc((Nverts+1)*sizeof(double));
    double* zz = (double*) malloc((Nverts+1)*sizeof(double));
    int* vr = (int*) malloc((Nverts+1)*sizeof(int));
    r = GmfGetBlock(id_mesh, GmfVertices, 1, Nverts, 0, nullptr, nullptr,
                    GmfDouble, xx+1, xx+Nverts+1,
                    GmfDouble, yy+1, yy+Nverts+1,
                    GmfDouble, zz+1, zz+Nverts+1,
                    GmfInt, vr+1, vr+Nverts+1);
    if (!r) {
        fprintf(stderr, "Echec de lecture des vertices\n");
        return -1;
    }
    int64_t* vv = (int64_t*) malloc(4*(Ntets+1)*sizeof(int64_t));
    int* tr = (int*) malloc((Ntets+1)*sizeof(int));
    r = GmfGetBlock(id_mesh, GmfTetrahedra, 1, Ntets, 0, nullptr, nullptr,
                    GmfLong, vv+0*Ntets+1, vv+1*Ntets,
                    GmfLong, vv+1*Ntets+1, vv+2*Ntets,
                    GmfLong, vv+2*Ntets+1, vv+3*Ntets,
                    GmfLong, vv+3*Ntets+1, vv+4*Ntets,
                    GmfInt, tr+1, tr+Ntets);
    if (!r) {
        fprintf(stderr, "Echec de lecture des tétraèdres\n");
        return -1;
    }
    GmfCloseMesh(id_mesh);
    Vertex* verticesCPU = new Vertex[Nverts];
    for (int i = 0; i < Nverts; i++){
        verticesCPU[i].x = xx[i+1];
        verticesCPU[i].y = yy[i+1];
        verticesCPU[i].z = zz[i+1];
    }
    Tetrahedron* tetsCPU = new Tetrahedron[Ntets];
    for (int i = 0; i < Ntets; i++){
        tetsCPU[i].v[0] = (int)(vv[i] - 1);
        tetsCPU[i].v[1] = (int)(vv[i+Ntets] - 1);
        tetsCPU[i].v[2] = (int)(vv[i+2*Ntets] - 1);
        tetsCPU[i].v[3] = (int)(vv[i+3*Ntets] - 1);
    }
    BoundingBox bbox = calculate_bounding_box(Nverts, xx, yy, zz);
    double xmin = bbox.xmin, xmax = bbox.xmax;
    double ymin = bbox.ymin, ymax = bbox.ymax;
    double zmin = bbox.zmin, zmax = bbox.zmax;
    // Pour la projection sur (y,z)
    double proj_min1 = ymin, proj_max1 = ymax;
    double proj_min2 = zmin, proj_max2 = zmax;
    printf("Maillage chargé : %ld vertices, %ld tétraèdres\n", Nverts, Ntets);

    // Chargement de la solution
    const char* solFile = "smallo.sol";
    int id_sol_version, id_sol_dim;
    int Nsol, Ntypes, SolSize, deg, nmbNod;
    int SolTypes[2];
    int64_t id_sol = GmfOpenMesh(solFile, GmfRead, &id_sol_version, &id_sol_dim);
    if (!id_sol) {
        fprintf(stderr, "Impossible d'ouvrir le fichier solution %s\n", solFile);
        return -1;
    }
    Nsol = GmfStatKwd(id_sol, GmfSolAtVertices, &Ntypes, &SolSize, SolTypes, &deg, &nmbNod);
    double* solData = (double*) malloc((Nsol+1)*sizeof(double));
    r = GmfGetBlock(id_sol, GmfSolAtVertices, 1, Nsol, 0, nullptr, nullptr,
                    GmfDouble, solData+1, solData+Nsol);
    if (!r) {
        fprintf(stderr, "Echec de lecture de la solution\n");
        return -1;
    }
    GmfCloseMesh(id_sol);
    printf("Solution chargée : %d valeurs\n", Nsol);

    // Conversion pour GPU
    struct FVertex { float x, y, z; };
    FVertex* verticesGPU = new FVertex[Nverts];
    for (int i = 0; i < Nverts; i++){
        verticesGPU[i].x = (float) verticesCPU[i].x;
        verticesGPU[i].y = (float) verticesCPU[i].y;
        verticesGPU[i].z = (float) verticesCPU[i].z;
    }
    struct FTetrahedron { int v[4]; };
    FTetrahedron* tetsGPU = new FTetrahedron[Ntets];
    for (int i = 0; i < Ntets; i++){
        tetsGPU[i].v[0] = tetsCPU[i].v[0];
        tetsGPU[i].v[1] = tetsCPU[i].v[1];
        tetsGPU[i].v[2] = tetsCPU[i].v[2];
        tetsGPU[i].v[3] = tetsCPU[i].v[3];
    }
    float* solGPU = new float[Nsol];
    for (int i = 0; i < Nsol; i++){
        solGPU[i] = (float) solData[i+1];
    }
    printf("Conversion vers données GPU terminée.\n");

    // Création des SSBO pour vertices, tétraèdres, solution
    GLuint ssboVertices, ssboTets, ssboSol;
    glGenBuffers(1, &ssboVertices);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVertices);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nverts*sizeof(FVertex), verticesGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVertices);

    glGenBuffers(1, &ssboTets);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTets);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Ntets*sizeof(FTetrahedron), tetsGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboTets);

    glGenBuffers(1, &ssboSol);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSol);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nsol*sizeof(float), solGPU, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboSol);

    // Construction de la grille (projection sur (y,z))
    int nCells_u = 100, nCells_v = 100;
    int totalCells = nCells_u * nCells_v;
    int* cellCounts = (int*) calloc(totalCells, sizeof(int));
    double projRange1 = proj_max1 - proj_min1;
    double projRange2 = proj_max2 - proj_min2;
    double cell_width = projRange1 / nCells_u;
    double cell_height = projRange2 / nCells_v;
    for (int i = 0; i < Ntets; i++){
        double p_u_min = 1e12, p_u_max = -1e12;
        double p_v_min = 1e12, p_v_max = -1e12;
        for (int j = 0; j < 4; j++){
            double u = verticesCPU[tetsCPU[i].v[j]].y;
            double v = verticesCPU[tetsCPU[i].v[j]].z;
            if(u < p_u_min) p_u_min = u;
            if(u > p_u_max) p_u_max = u;
            if(v < p_v_min) p_v_min = v;
            if(v > p_v_max) p_v_max = v;
        }
        int cell_u_min = (int)((p_u_min - proj_min1) / cell_width);
        int cell_u_max = (int)((p_u_max - proj_min1) / cell_width);
        int cell_v_min = (int)((p_v_min - proj_min2) / cell_height);
        int cell_v_max = (int)((p_v_max - proj_min2) / cell_height);
        if(cell_u_min < 0) cell_u_min = 0;
        if(cell_u_max >= nCells_u) cell_u_max = nCells_u - 1;
        if(cell_v_min < 0) cell_v_min = 0;
        if(cell_v_max >= nCells_v) cell_v_max = nCells_v - 1;
        for (int r = cell_u_min; r <= cell_u_max; r++){
            for (int c = cell_v_min; c <= cell_v_max; c++){
                int cellIndex = r * nCells_v + c;
                cellCounts[cellIndex]++;
            }
        }
    }
    GridCell* gridCells = (GridCell*) malloc(totalCells*sizeof(GridCell));
    int totalIndices = 0;
    for (int i = 0; i < totalCells; i++){
        gridCells[i].offset = totalIndices;
        gridCells[i].count = cellCounts[i];
        totalIndices += cellCounts[i];
    }
    free(cellCounts);
    int* gridIndices = (int*) malloc(totalIndices*sizeof(int));
    int* cellOffsets = (int*) calloc(totalCells, sizeof(int));
    for (int i = 0; i < Ntets; i++){
        double p_u_min = 1e12, p_u_max = -1e12;
        double p_v_min = 1e12, p_v_max = -1e12;
        for (int j = 0; j < 4; j++){
            double u = verticesCPU[tetsCPU[i].v[j]].y;
            double v = verticesCPU[tetsCPU[i].v[j]].z;
            if(u < p_u_min) p_u_min = u;
            if(u > p_u_max) p_u_max = u;
            if(v < p_v_min) p_v_min = v;
            if(v > p_v_max) p_v_max = v;
        }
        int cell_u_min = (int)((p_u_min - proj_min1) / cell_width);
        int cell_u_max = (int)((p_u_max - proj_min1) / cell_width);
        int cell_v_min = (int)((p_v_min - proj_min2) / cell_height);
        int cell_v_max = (int)((p_v_max - proj_min2) / cell_height);
        if(cell_u_min < 0) cell_u_min = 0;
        if(cell_u_max >= nCells_u) cell_u_max = nCells_u - 1;
        if(cell_v_min < 0) cell_v_min = 0;
        if(cell_v_max >= nCells_v) cell_v_max = nCells_v - 1;
        for (int r = cell_u_min; r <= cell_u_max; r++){
            for (int c = cell_v_min; c <= cell_v_max; c++){
                int cellIndex = r * nCells_v + c;
                int pos = gridCells[cellIndex].offset + cellOffsets[cellIndex];
                gridIndices[pos] = i;
                cellOffsets[cellIndex]++;
            }
        }
    }
    free(cellOffsets);
    printf("Grille construite : %d cellules, %d indices\n", totalCells, totalIndices);

    // Création des SSBO pour la grille
    GLuint ssboGridCells, ssboGridIndices;
    glGenBuffers(1, &ssboGridCells);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGridCells);
    glBufferData(GL_SHADER_STORAGE_BUFFER, totalCells*sizeof(GridCell), gridCells, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboGridCells);

    glGenBuffers(1, &ssboGridIndices);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGridIndices);
    glBufferData(GL_SHADER_STORAGE_BUFFER, totalIndices*sizeof(int), gridIndices, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssboGridIndices);
    free(gridCells);
    free(gridIndices);

    // Calcul de u_maxAbs (pour la coloration)
    float u_maxAbs = 0.0f;
    {
        float minVal = solGPU[0], maxVal = solGPU[0];
        for (int i = 1; i < Nsol; i++){
            if(solGPU[i] < minVal) minVal = solGPU[i];
            if(solGPU[i] > maxVal) maxVal = solGPU[i];
        }
        u_maxAbs = fabs(minVal) > fabs(maxVal) ? fabs(minVal) : fabs(maxVal);
    }
    printf("u_maxAbs = %f\n", u_maxAbs);

    // ----------------- Création d'un FBO et d'une texture de sortie -----------------
    GLuint fbo, outputTex;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &outputTex);
    glBindTexture(GL_TEXTURE_2D, outputTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer incomplet !\n");
        return -1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Création du programme de rendu (quad plein écran)
    GLuint renderProgram = createProgram(quadVertexShaderSource, quadFragmentShaderSource);

    // Création du programme compute 
    GLuint computeProgram = createComputeProgram(computeShaderSource);
    glUseProgram(computeProgram);
    glUniform2i(glGetUniformLocation(computeProgram, "imageSize"), width, height);
    glUniform1i(glGetUniformLocation(computeProgram, "nCells_u"), nCells_u);
    glUniform1i(glGetUniformLocation(computeProgram, "nCells_v"), nCells_v);
    glUniform1f(glGetUniformLocation(computeProgram, "projMin1"), (float)proj_min1);
    glUniform1f(glGetUniformLocation(computeProgram, "projMax1"), (float)proj_max1);
    glUniform1f(glGetUniformLocation(computeProgram, "projMin2"), (float)proj_min2);
    glUniform1f(glGetUniformLocation(computeProgram, "projMax2"), (float)proj_max2);
    glUniform1f(glGetUniformLocation(computeProgram, "u_maxAbs"), u_maxAbs);
    int planeAxis = PLANE_X;
    glUniform1i(glGetUniformLocation(computeProgram, "planeAxis"), planeAxis);

    // Création d'un quad plein écran pour le rendu
    float quadVertices[] = { -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // ----------------- Boucle de rendu pour produire le film -----------------
    int frameIndex = 0;
    double totalComputeTime = 0.0;
    // On fait varier planeVal de (xmin+1) à xmax avec un incrément de 1.0

    bool windowShown = false;
    for (float pv = (float)xmin + 1.0f; pv <= (float)xmax; pv += 1.0f) {
        double tStart = millitime();
        // Mise à jour de la valeur du plan
        glUseProgram(computeProgram);
        glUniform1f(glGetUniformLocation(computeProgram, "planeVal"), pv);
        // Lier la texture de sortie pour le compute shader
        glBindImageTexture(0, outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        int groupCountX = (width + 15) / 16;
        int groupCountY = (height + 15) / 16;
        glDispatchCompute(groupCountX, groupCountY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	double tComputeEnd = millitime();
	double tCompute = tComputeEnd - tStart;
	totalComputeTime += tCompute;
        
	// We display the window that was hidden
	if (!windowShown){
	    glfwShowWindow(window);
	    windowShown = true;
	}

	/*
        // Rendu du quad sur la fenêtre (affichage)
        if (window == nullptr) {
            // Créer la fenêtre maintenant que les premières images sont prêtes
            window = glfwCreateWindow(width, height, "Mesh Compute Film", nullptr, nullptr);
            if (!window){
                fprintf(stderr, "Echec de la création de la fenêtre\n");
                glfwTerminate();
                return -1;
            }
            glfwMakeContextCurrent(window);
            glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
            if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
                fprintf(stderr, "Echec de l'initialisation de GLAD\n");
                return -1;
            }
            glViewport(0, 0, width, height);
        }
        */
        // Rendu du quad sur la fenêtre (affichage)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(renderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, outputTex);
        glUniform1i(glGetUniformLocation(renderProgram, "computedTexture"), 0);
        glBindVertexArray(quadVAO);

	double tRenderStart = millitime();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glfwSwapBuffers(window);
        glfwPollEvents();
	double tRenderEnd = millitime();
	double tRender = tRenderEnd - tRenderStart;

	double tFrame = millitime() - tStart;
        printf("Frame %d: planeVal = %.2f | Compute = %.3f ms |Render = %.3f ms | Total = %.3f ms\n", frameIndex, pv, tCompute, tRender, tFrame);
        frameIndex++;
    }
    printf("Temps total compute shader rendering (compute uniquement) : %.3f ms\n", totalComputeTime);

    // Nettoyage
    glDeleteProgram(computeProgram);
    glDeleteProgram(renderProgram);
    glDeleteBuffers(1, &ssboVertices);
    glDeleteBuffers(1, &ssboTets);
    glDeleteBuffers(1, &ssboSol);
    glDeleteBuffers(1, &ssboGridCells);
    glDeleteBuffers(1, &ssboGridIndices);
    glDeleteBuffers(1, &quadVBO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &outputTex);
    glfwDestroyWindow(window);
    glfwTerminate();

    free(xx); free(yy); free(zz); free(vr); free(vv); free(tr);
    delete[] verticesCPU;
    delete[] tetsCPU;
    free(solData);
    delete[] verticesGPU;
    delete[] tetsGPU;
    delete[] solGPU;

    return 0;
}

