#ifdef _WIN32
  // Windows n’a pas <sys/time.h> : on bascule sur chrono pour gettimeofday()
  #include <chrono>
  inline double get_time_seconds() {
    using namespace std::chrono;
    return duration<double>(high_resolution_clock::now().time_since_epoch()).count();
  }
#else
  // Sur Unix, on garde sys/time.h et gettimeofday()
  #include <sys/time.h>
  inline double get_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }
#endif


// Millisecond timestamp, wrapper portable
inline double millitime() {
  return 1000.0 * get_time_seconds();
}

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <algorithm>
#include <string>
#include <atomic>
#include <thread>
#include <iostream>
#include <cstring>
#include <cinttypes>
#include <errno.h>
#include <fstream>



// Définition des axes
#define PLANE_X 0
#define PLANE_Y 1
#define PLANE_Z 2

// Measuring the time in milliseconds

// Saving in PPM format (reading the framebuffer with glReadPixels)
void savePPM(const char* filename, int width, int height) {
    unsigned char* pixels = (unsigned char*) malloc(width * height * 3);
    if (!pixels) {
        fprintf(stderr, "[ERROR] Allocation failed for pixels.\n");
        return;
    }
    // Getting the picture from the current buffer
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // Saving to a PPM file
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open file %s for writing.\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int j = height - 1; j >= 0; j--) {
        fwrite(pixels + j * width * 3, 1, width * 3, fp);
    }
    fclose(fp);
    free(pixels);
    printf("[INFO] Image saved in %s\n", filename);
}


// Callback d'erreur GLFW
void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d): %s\n", error, description);
}

extern "C" {
    #include "libmeshb7.h"
}


// ===============================================
//    Compute Shader compiler function
// ===============================================
GLuint compileComputeShader(const char* source) {
    // Create an unsigned int id associated to the compute shader
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    // Sprecify the GLSL code to associate to our shader in source
    glShaderSource(shader, 1, &source, nullptr);
    // Compiling : checking syntax, GLSL -> GPU bytecode, etc.
    glCompileShader(shader);
    GLint success;
    // Verification with the log
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512] = {0};
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "[ERROR] Compute shader compilation failed: %s\n", infoLog);
    }
    return shader;
}
// ===============================================
//    Compute Shader linker function
// ===============================================
GLuint createComputeProgram(const char* csSource) {
    // Calling the precedent function to compile the compute shader GLSL source code	
    GLuint ccs = compileComputeShader(csSource);
    // program allows us to execute the shader on the GPU
    GLuint program = glCreateProgram();
    // Attaching the compiled compute shader to the program
    glAttachShader(program, ccs);
    // Linking the program, our compute shader will be executable on the GPU
    glLinkProgram(program);
    GLint success;
    // Verification with the log
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512] = {0};
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "[ERROR] Compute program linking failed: %s\n", infoLog);
    }
    glDeleteShader(ccs);
    return program;
}

// ======================================================
// Structures et fonctions pour le maillage côté CPU
// ======================================================
typedef struct {
    double x, y, z;
} Vertex;

typedef struct {
    int v[4];
} Tetrahedron;

typedef struct {
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
} BoundingBox;

typedef struct {
    int offset;
    int count;
} GridCell;

BoundingBox calculate_bounding_box(int Nverts, double *xx, double *yy, double *zz) {
    BoundingBox bbox;
    if (Nverts < 1) {
        fprintf(stderr, "[ERROR] No vertices for computing the bounding box.\n");
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
// ================================================================================
// Fonction pour vérifier si un tétraèdre est dégénéré
// ================================================================================
bool is_degenerate_tetrahedron(Tetrahedron tet, Vertex *vertices, int Nverts) {
    for (int j = 0; j < 4; j++) {
        if (tet.v[j] < 0 || tet.v[j] >= Nverts) {
            return true;
        }
    }
    Vertex v0 = vertices[tet.v[0]];
    Vertex v1 = vertices[tet.v[1]];
    Vertex v2 = vertices[tet.v[2]];
    Vertex v3 = vertices[tet.v[3]];
    double v0x = v1.x - v0.x;
    double v0y = v1.y - v0.y;
    double v0z = v1.z - v0.z;
    double v1x = v2.x - v0.x;
    double v1y = v2.y - v0.y;
    double v1z = v2.z - v0.z;
    double v2x = v3.x - v0.x;
    double v2y = v3.y - v0.y;
    double v2z = v3.z - v0.z;
    double detT = v0x * (v1y * v2z - v1z * v2y) -
                  v0y * (v1x * v2z - v1z * v2x) +
                  v0z * (v1x * v2y - v1y * v2x);
    return fabs(detT) < 1e-6;
}

// ================================================================================
// Fonction pour filtrer les tétraèdres intersectant un plan
// ================================================================================
int filter_tetrahedrons_parallel(int Ntet, Tetrahedron *tets, Vertex *vertices, int Nverts, int axis, double plane_val, int **filtered_tets, int *filtered_count) {
    if (axis != PLANE_X && axis != PLANE_Y && axis != PLANE_Z) {
        fprintf(stderr, "[ERROR] Invalid axis chosen. Program will exit.\n");
        return EXIT_FAILURE;
    }
    bool *flags = (bool*) calloc(Ntet, sizeof(bool));
    if (!flags) {
        fprintf(stderr, "[ERROR] Memory allocation failed for flags.\n");
        return EXIT_FAILURE;
    }
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < Ntet; i++) {
        if (is_degenerate_tetrahedron(tets[i], vertices, Nverts)) {
            continue;
        }
        bool intersect = false;
        int has_low = 0, has_high = 0;
        for (int j = 0; j < 4; j++) {
            double coord;
            switch (axis) {
                case PLANE_X: coord = vertices[tets[i].v[j]].x; break;
                case PLANE_Y: coord = vertices[tets[i].v[j]].y; break;
                case PLANE_Z: coord = vertices[tets[i].v[j]].z; break;
            }
	    // cast to float to emulate what I suspect the GPU is doing --MP
            if ((float)coord <= (float)plane_val) has_low = 1;
            if ((float)coord >= (float)plane_val) has_high = 1;
            if (has_low && has_high) {
                intersect = true;
                break;
            }
        }
        if (intersect) {
            flags[i] = true;
            count++;
        }
    }
    *filtered_count = count;
    *filtered_tets = (int*) malloc(count * sizeof(int));
    if (!(*filtered_tets)) {
        fprintf(stderr, "[ERROR] Memory allocation failure for filtered_tets.\n");
        free(flags);
        return EXIT_FAILURE;
    }
    int pos = 0;
    for (int i = 0; i < Ntet; i++) {
        if (flags[i]) {
            (*filtered_tets)[pos++] = i;
        }
    }
    free(flags);
    return 1;
}
void checkSSBOSize(GLenum target, GLsizeiptr expectedSize) {
    GLint size = 0;
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &size);
    if (size != expectedSize)
        printf("[ERROR] SSBO size mismatch: expected %ld, got %d\n", expectedSize, size);
    else
        printf("[DEBUG] SSBO size is correct (%d bytes).\n", size);
}
// =========================================
// Compute shader to filter the tetrahedra
// =========================================
const char* FilterComputeShaderSource = R"(
#version 450

layout(local_size_x = 1024) in;

uniform int Nverts;
uniform int Ntet;
uniform int axis;
uniform float planeVal;

layout( binding = 0) buffer VertexBuffer { vec3 vertices[]; };
layout( binding = 1) buffer TetrahedronBuffer { ivec4 tets[]; };

layout( binding = 6) buffer FilteredIndices {
    int filteredIndices[];
};

layout(binding = 7) uniform atomic_uint filteredCounter;

double det3(vec3 a, vec3 b, vec3 c) {
    return dot(a, cross(b, c));
}

bool isDegenerate(ivec4 tet){
    // 1) Vérification des indices hors bornes
    for (int j = 0; j < 4; ++j) {
        if (tet[j] < 0 || tet[j] >= Nverts) {
            return true;
        }
    }
    vec3 p0 = vertices[tet.x];
    vec3 p1 = vertices[tet.y];
    vec3 p2 = vertices[tet.z];
    vec3 p3 = vertices[tet.w];
    // same tolerance than the CPU
    return abs(det3(p1 - p0, p2 - p0, p3 - p0)) < 1e-6;
}

bool tetrahedronIntersects(ivec4 tet) {
    bool low = false;
    bool high = false;
    for (int i = 0; i < 4; i++) {
        float coord;
        if (axis == 0) coord = vertices[tet[i]].x;
        else if (axis == 1) coord = vertices[tet[i]].y;
        else coord = vertices[tet[i]].z;
        if (coord <= planeVal) low = true;
        if (coord >= planeVal) high = true;
    }
    return low && high;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(Ntet)) return;

    ivec4 tet = tets[i];

    if (isDegenerate(tet)){
        return;         // We avoid the degenerate tet
    }
    if (tetrahedronIntersects(tet)) {
        uint pos = atomicCounterIncrement(filteredCounter);
        filteredIndices[pos] = int(i);
    }
}
)";



// ================================================
// Compute shader for the off-screen rendering
// ================================================
const char* computeShaderSource = R"(
#version 450
// specifying the size of the local workgroup, 256 invocations/threads per workgroup
layout (local_size_x = 16, local_size_y = 16) in;
// 
// Uniform : variables declared in the shader that conserve a unique constant value
// during all the execution of the compute shader. All the threads/invocation access 
// to the same value for the current frame. 
// 
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
//
// Structures for the SSBO
//
struct FVertex { float x, y, z; };
struct FTetrahedron { ivec4 v; };
layout(std430, binding = 0) buffer VertexBuffer { FVertex vertices[]; };
layout(std430, binding = 1) buffer TetrahedronBuffer { FTetrahedron tets[]; };
layout(std430, binding = 2) buffer SolutionBuffer { float sol[]; };
//
// offset is the starting index in gridIndices[]
// count is the number of tetrahedra located in the cell
struct GridCell { int offset; int count; };
//
// gridCells[] is the vector of all the cells in the grid
// gridIndices[] store the tetrahedra indexes, grouped by cells
// Accessing the filtered tetrahdedra associated to the current plane val
// and browsing only those in the cell assocaited to our pixel.
layout(std430, binding = 4) buffer GridCellsBuffer { GridCell gridCells[]; };
layout(std430, binding = 5) buffer GridIndicesBuffer { int gridIndices[]; };
//
// Image output, uniform in writeonly mode, every thread will write its value in it
//
layout(rgba8, binding = 0) uniform writeonly image2D destImage;
// 
// Computing the determinant
//
float det3(vec3 a, vec3 b, vec3 c) { return dot(a, cross(b, c)); }
//
// Computing the barycentric coordinates
//
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
    // 
    // Compute the pixel position of the current thread/invocation in the total grid
    //
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    //
    // Verification
    //
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;
    //
    // Normalising between [0;1] for each dimension, easier to manipulate the data
    // like this because it is independant from the resolution
    //
    vec2 uv = vec2(pixelCoord) / vec2(imageSize);
    float coord1 = mix(projMin1, projMax1, uv.x);
    float coord2 = mix(projMin2, projMax2, uv.y);
    // 
    // Setting a coordinate with respect to the cut plane
    //
    vec3 p;
    if (planeAxis == 0) p = vec3(planeVal, coord1, coord2);
    else if (planeAxis == 1) p = vec3(coord1, planeVal, coord2);
    else p = vec3(coord1, coord2, planeVal);
    // 
    // Research & interpolation
    //
    float interpolated = 0.0;
    bool found = false;
    vec4 bary;
    // Here, we use the grid to limit the number of tetrahedra we test
    int cell_u = int((coord1 - projMin1) / ((projMax1 - projMin1) / float(nCells_u)));
    int cell_v = int((coord2 - projMin2) / ((projMax2 - projMin2) / float(nCells_v)));
    cell_u = clamp(cell_u, 0, nCells_u - 1);
    cell_v = clamp(cell_v, 0, nCells_v - 1);
    int cellIndex = cell_u * nCells_v + cell_v;
    GridCell cell = gridCells[cellIndex];
    //
    // We browse only the tetrahedra we are interested in
    // and we interpolate the value 
    for (int i = 0; i < cell.count; i++){
        int tetIndex = gridIndices[cell.offset + i];
        FTetrahedron tet = tets[tetIndex];
        if (computeBarycentric(p, tet, bary)) {
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

// the easy way to detect OpenGL errors
// https://www.khronos.org/opengl/wiki/OpenGL_Error
void GLAPIENTRY
MessageCallback( GLenum source,
		 GLenum type,
		 GLuint id,
		 GLenum severity,
		 GLsizei length,
		 const GLchar* message,
                 const void* userParam )
{
  fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
	   ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
	   type, severity, message );
}
    

// ====================================================================
// MAIN
// ====================================================================
int main(int argc, char **argv) {
    const int width = 1024, height = 1024;
    const size_t bufferSize = width * height * 3; // 3 octets par pixel (RGB)
    int nCells_u = 250, nCells_v = 250; 

    // =============================================================
    // Initialisation of glfw and creation of the off-screen window
    // =============================================================
    setvbuf(stdout, nullptr, _IONBF, 0);
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()){
        fprintf(stderr, "[ERROR] Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }


    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Off-screen
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(width, height, "Off-screen Renderer", nullptr, nullptr);
    // Verification
    if (!window) {
        fprintf(stderr, "[ERROR] Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    // Making the OpenGL context associated to the window 
    // the current context for the runnning thread.
    glfwMakeContextCurrent(window);
    // Definition of the vsync (vertical synchronisation) interval.
    // Putting it to zero means we deactivate it, so we don't wait the
    // screen to refresh, useful for off-screen rendering.
    glfwSwapInterval(0);
    // Verification
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        fprintf(stderr, "[ERROR] Failed to initialize GLAD\n");
        return EXIT_FAILURE;
    }
    glViewport(0, 0, width, height);
 
    // During init, enable debug output
    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );
    
    // =================================
    // Reading the mesh file .mesh
    // =================================
    double t0 = millitime();
    int r;
    int64_t id_mesh, Nverts, Ntets;
    const char *meshFile = "animal.mesh";
    int id_mesh_version, id_mesh_dim;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    printf("[INFO] Opening mesh\n");
    id_mesh = GmfOpenMesh(meshFile, GmfRead, &id_mesh_version, &id_mesh_dim);
    if (!id_mesh) { fprintf(stderr, "[ERROR] Cannot open '%s'\n", meshFile); return EXIT_FAILURE; }
    Nverts = GmfStatKwd(id_mesh, GmfVertices);
    Ntets  = GmfStatKwd(id_mesh, GmfTetrahedra);
    printf("[INFO] Mesh with %ld vertices and %ld tetrahedra\n", Nverts, Ntets);
    double *xx = (double*) malloc((Nverts+1) * sizeof(double));
    double *yy = (double*) malloc((Nverts+1) * sizeof(double));
    double *zz = (double*) malloc((Nverts+1) * sizeof(double));
    int *vr = (int*) malloc((Nverts+1) * sizeof(int));
    r = GmfGetBlock(id_mesh, GmfVertices, 1, Nverts, 0, nullptr, nullptr,
                    GmfDouble, xx+1, xx+Nverts+1,
                    GmfDouble, yy+1, yy+Nverts+1,
                    GmfDouble, zz+1, zz+Nverts+1,
                    GmfInt, vr+1, vr+Nverts+1);
    if (!r) {
        fprintf(stderr, "[ERROR] Reading vertices failed\n");
        free(xx); free(yy); free(zz); free(vr);
        GmfCloseMesh(id_mesh);
        return EXIT_FAILURE;
    }
    int64_t *vv = (int64_t*) malloc(4*(Ntets+1) * sizeof(int64_t));
    int *tr = (int*) malloc((Ntets+1) * sizeof(int));
    r = GmfGetBlock(id_mesh, GmfTetrahedra, 1, Ntets, 0, nullptr, nullptr,
                    GmfLong, vv+0*Ntets+1, vv+1*Ntets,
                    GmfLong, vv+1*Ntets+1, vv+2*Ntets,
                    GmfLong, vv+2*Ntets+1, vv+3*Ntets,
                    GmfLong, vv+3*Ntets+1, vv+4*Ntets,
                    GmfInt, tr+1, tr+Ntets);
    if (!r) {
        fprintf(stderr, "[ERROR] Reading tetrahedra failed\n");
        free(xx); free(yy); free(zz); free(vr); free(vv); free(tr);
        GmfCloseMesh(id_mesh);
        return EXIT_FAILURE;
    }
    GmfCloseMesh(id_mesh);
    Vertex *verticesCPU = new Vertex[Nverts];
    for (int i = 0; i < Nverts; i++) {
        verticesCPU[i].x = xx[i+1];
        verticesCPU[i].y = yy[i+1];
        verticesCPU[i].z = zz[i+1];
    }
    Tetrahedron *tetsCPU = new Tetrahedron[Ntets];
    for (int i = 0; i < Ntets; i++) {
    	// vv has unit-offset indexing and unit-offset content;
	// tetsCPU has zero-offset for both
        tetsCPU[i].v[0] = (int)(vv[1 + i] - 1);
        tetsCPU[i].v[1] = (int)(vv[1 + i + Ntets] - 1);
        tetsCPU[i].v[2] = (int)(vv[1 + i + 2*Ntets] - 1);
        tetsCPU[i].v[3] = (int)(vv[1 + i + 3*Ntets] - 1);
    }
    BoundingBox bbox = calculate_bounding_box(Nverts, xx, yy, zz);
    xmin = bbox.xmin; xmax = bbox.xmax;
    ymin = bbox.ymin; ymax = bbox.ymax;
    zmin = bbox.zmin; zmax = bbox.zmax;
    double proj_min1 = ymin, proj_max1 = ymax;
    double proj_min2 = zmin, proj_max2 = zmax;
    printf("[INFO] Mesh loaded in %.3f seconds\n", 0.001*(millitime()-t0));

    // ===========================================
    // Reading the solution file .sol
    // ===========================================
    t0 = millitime();
    const char *solFile = "animal.sol";
    int id_sol_version, id_sol_dim;
    int Nsol, Ntypes, SolSize, deg, nmbNod;
    int SolTypes[2];
    printf("[INFO] Opening solution file\n");
    int64_t id_sol = GmfOpenMesh(solFile, GmfRead, &id_sol_version, &id_sol_dim);
    if (!id_sol) { fprintf(stderr, "[ERROR] Cannot open '%s'\n", solFile); return EXIT_FAILURE; }
    Nsol = GmfStatKwd(id_sol, GmfSolAtVertices, &Ntypes, &SolSize, SolTypes, &deg, &nmbNod);
    double *solData = (double*) malloc((Nsol+1) * sizeof(double));
    r = GmfGetBlock(id_sol, GmfSolAtVertices, 1, Nsol, 0, nullptr, nullptr,
                    GmfDouble, solData+1, solData+Nsol);
    if (!r) {
        fprintf(stderr, "[ERROR] Reading solution failed\n");
        free(solData);
        GmfCloseMesh(id_sol);
        return EXIT_FAILURE;
    }
    GmfCloseMesh(id_sol);
    printf("[INFO] Mesh and solution loaded: %ld vertices, %ld tetrahedra, %d solution values\n", Nverts, Ntets, Nsol);
    printf("[INFO] Solution loaded in %.3f seconds\n", 0.001*(millitime()-t0));

    // ==========================================================
    // Converting CPU -> GPU optimizing the memory and treatment
    // ==========================================================
    t0 = millitime();

    // 4 bytes for padding
    struct FVertex { 
	    float x, y, z;
	    float pad;
    };
    FVertex* verticesGPU = new FVertex[Nverts];
    for (int i = 0; i < Nverts; i++) {
        verticesGPU[i].x = (float) verticesCPU[i].x;
        verticesGPU[i].y = (float) verticesCPU[i].y;
        verticesGPU[i].z = (float) verticesCPU[i].z;
    }
    struct FTetrahedron { int v[4]; };
    FTetrahedron* tetsGPU = new FTetrahedron[Ntets];
    for (int i = 0; i < Ntets; i++) {
        tetsGPU[i].v[0] = tetsCPU[i].v[0];
        tetsGPU[i].v[1] = tetsCPU[i].v[1];
        tetsGPU[i].v[2] = tetsCPU[i].v[2];
        tetsGPU[i].v[3] = tetsCPU[i].v[3];
    }
    float* solGPU = new float[Nsol];
    for (int i = 0; i < Nsol; i++) {
        solGPU[i] = (float) solData[i+1];
    }
    printf("[INFO] Conversion to GPU data completed in %.3f seconds\n", 0.001*(millitime()-t0));

//=======================================================================================
//
//=======================================================================================
// Affiche les indices du tétra 0
    printf("Tetrahedron 0 vertex indices: [%d, %d, %d, %d]\n",
		    tetsCPU[0].v[0],
		    tetsCPU[0].v[1],
		    tetsCPU[0].v[2],
		    tetsCPU[0].v[3]);
    // Vérifie qu’ils sont tous dans [0, Nverts)
    for (int j = 0; j < 4; ++j) {
	    int idx = tetsCPU[0].v[j];
	    if (idx < 0 || idx >= Nverts)
		    printf("  -> index %d is OUT OF BOUNDS! (Nverts=%ld)\n", idx, Nverts);
	    else
		    printf("  -> vertex[%d] is valid\n", idx);
    }

    // ====================================
    // Creation of the SSBO for the GPU
    // ====================================
    GLuint ssboVertices, ssboTets, ssboSol, ssboFilteredIndices, atomicCounterBuffer;

    // 1) Vertices
    // Generate an id for the buffer used for the vertices.
    glGenBuffers(1, &ssboVertices);

    // Bind the buffer to the target point GL_SHADER_STORAGE_BUFFER, so that 
    // it can be used as a SSBO.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVertices);

    // Allocating and filling the buffer with the verticesGPU data
    // GL_STATIC_DRAW indicates the data won't change often
    // we want to read them mainly.
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nverts * sizeof(FVertex), verticesGPU, GL_STATIC_DRAW);
    // Bind our buffer to the specific binding point 0. 
    // Shaders will be able to access the data thanks to this.
    // This buffer allow the shaders to directly access the GPU memory data.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVertices);

    // 2) Tetrahedra.
    glGenBuffers(1, &ssboTets);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTets);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Ntets * sizeof(FTetrahedron), tetsGPU, GL_STATIC_DRAW);
    // Binding point 1
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboTets);
    
    // 3) Solution
    glGenBuffers(1, &ssboSol);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSol);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nsol * sizeof(float), solGPU, GL_STATIC_DRAW);
    // Binding point 2
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboSol);

    // 4) filteredIndices
    glGenBuffers(1, &ssboFilteredIndices);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboFilteredIndices);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Ntets * sizeof(GLint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssboFilteredIndices);

    // 5) atomic counter
    glGenBuffers(1, &atomicCounterBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 7, atomicCounterBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    printf("##0\n");
    if(errno) printf("ERROR: %s", strerror(errno)), exit(1);



    // Mesure du filtrage GPU
    double t0_gpu = millitime();
    
    // Remise à zéro du compteur
    GLuint zero = 0;
    double pv = 301.0000005;

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    // Dispatch
    GLuint filterProgram = createComputeProgram(FilterComputeShaderSource);
    glUseProgram(filterProgram);

    glUniform1i(glGetUniformLocation(filterProgram, "Nverts"), Nverts);
    glUniform1i(glGetUniformLocation(filterProgram, "Ntet"), (GLint)Ntets);
    glUniform1i(glGetUniformLocation(filterProgram, "axis"), PLANE_X);
    glUniform1f(glGetUniformLocation(filterProgram, "planeVal"), pv);

    glDispatchCompute((GLuint)((Ntets + 1023) / 1024), 1, 1);
    //glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT |
    //                GL_SHADER_STORAGE_BARRIER_BIT);

    // 4.c) Lecture du résultat GPU
    GLuint gpuCount = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0,
                       sizeof(GLuint), &gpuCount);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    std::vector<int> gpuFiltered(gpuCount);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboFilteredIndices);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       gpuCount * sizeof(int),
                       gpuFiltered.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    double t1_gpu = millitime();
    printf("[TIMING] GPU filter (incl. read‑back): %.3f ms\n", t1_gpu - t0_gpu);


    double t0_cpu = millitime();

    int cpuCount = 0;
    int* cpuFiltered = nullptr;

    filter_tetrahedrons_parallel(
		    (int)Ntets,
		    tetsCPU,
		    verticesCPU,
		    Nverts,
		    PLANE_X,
		    pv,
		    &cpuFiltered,
		    &cpuCount
    );
    double t1_cpu = millitime();

    printf("[TIMING] CPU filter: %.3f ms\n", t1_cpu - t0_cpu);

    int idx = 0;
    bool cpuDeg = is_degenerate_tetrahedron(tetsCPU[idx], verticesCPU, Nverts);
    bool low=false, high=false;
    for(int j=0;j<4;j++){
	    double c = verticesCPU[tetsCPU[idx].v[j]].x;
	    if(c <= pv) low  = true;
	    if(c >= pv) high = true;
    }
    bool cpuInter = low && high;
    printf("Tet0 CPU: deg=%d  inter=%d\n", (int)cpuDeg, (int)cpuInter);

    printf("GPU  count = %u\n", gpuCount);
    printf("CPU  count = %d\n", cpuCount);

 
    // gpuFiltered : std::vector<int> de taille gpuCount
    // cpuFiltered : pointeur int* de taille cpuCount

 
    // … après le tri …
    std::vector<int> cpuList(cpuFiltered, cpuFiltered + cpuCount);
    std::sort(cpuList.begin(), cpuList.end());
    std::sort(gpuFiltered.begin(), gpuFiltered.end());

    // 1) Cherche les CPU‑only et GPU‑only
    std::vector<int> onlyCPU, onlyGPU;

    // Tous les éléments CPU qui n'existent pas côté GPU
    for (int idx : cpuList) {
	    if (!std::binary_search(gpuFiltered.begin(), gpuFiltered.end(), idx)) {
		    onlyCPU.push_back(idx);
	    }
    }
    // Tous les éléments GPU qui n'existent pas côté CPU
    for (int idx : gpuFiltered) {
	    if (!std::binary_search(cpuList.begin(), cpuList.end(), idx)) {
		    onlyGPU.push_back(idx);
	    }
    }

    // 2) Si on a des différences, on exporte et on affiche "NO",
    //    sinon on affiche simplement "YES"
    if (onlyCPU.empty() && onlyGPU.empty()) {
	    // Tout correspond
	    printf("Lists match? YES\n");
    } else {
	    // Il y a des différences : on écrit le fichier et on affiche NO
	    std::ofstream out("diff_tets.txt");
	    out << "Only in CPU (" << onlyCPU.size() << "):\n";
	    for (int i = 0; i < (int)onlyCPU.size(); ++i) {
		    out << onlyCPU[i]
			    << ((i+1)%500 == 0 ? "\n" : " ");
	    }
	    out << "\n\nOnly in GPU (" << onlyGPU.size() << "):\n";
	    for (int i = 0; i < (int)onlyGPU.size(); ++i) {
 	      int j = onlyGPU[i];
	      out << onlyGPU[i]
		  << " " << verticesCPU[tetsCPU[j].v[0]].x
		  << " " << verticesCPU[tetsCPU[j].v[1]].x
		  << " " << verticesCPU[tetsCPU[j].v[2]].x
		  << " " << verticesCPU[tetsCPU[j].v[3]].x
		  << " " << "\n";
	    }
	    out.close();

	    printf("Lists match? NO\n");
	    std::cout << "Mismatch exported to diff_tets.txt\n";
    }

    free(cpuFiltered);
    glDeleteProgram(filterProgram);

    glDeleteBuffers(1, &ssboVertices);
    glDeleteBuffers(1, &ssboTets);
    glDeleteBuffers(1, &ssboSol);
    if(errno) printf("ERROR location 10: %s", strerror(errno)), exit(1);

    glfwDestroyWindow(window);
    glfwTerminate();
    
    free(xx); free(yy); free(zz); free(vr); free(vv); free(tr);
    delete[] verticesCPU;
    delete[] tetsCPU;
    free(solData);
    delete[] verticesGPU;
    delete[] tetsGPU;
    delete[] solGPU;

    return EXIT_SUCCESS;
}




