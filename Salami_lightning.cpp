#include <cstdio>        // printf, fprintf
#include <cstdlib>       // malloc, free, exit
#include <cstdint>       // fixed-size integers (int64_t)
#include <cmath>         // fabs, sqrt, INFINITY, mix
#ifdef _OPENMP
#include <omp.h>         // OpenMP pragmas 
#endif
#include <glad/glad.h>   // Loading OpenGL functions
#include <GLFW/glfw3.h>  // Creating the context and window (off screen)
#include <sys/time.h>    // timer
#include <cstring>       // strerror
#include <cinttypes>     // PRID64 to print int64_t
#include <errno.h>       // To print the OpenGL errors

extern "C" {
    #include "libmeshb7.h"  // API pour lire .mesh et .sol (Gmsh Mesh API)
}

// Definition of axes
#define PLANE_X 0
#define PLANE_Y 1
#define PLANE_Z 2

// Measuring the time in milliseconds
double millitime() {
    struct timeval tp;
    if(gettimeofday(&tp, nullptr))
        return 0;
    return 1000.0 * tp.tv_sec + 0.001 * tp.tv_usec;
}

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


// GLFW error callback
void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d): %s\n", error, description);
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
// Structures & functions for CPU-side mesh handling
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
// Check if a tetrahedron is degenerate
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
// Filter tetrahedra intersecting a given plane
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
            if (coord <= plane_val) has_low = 1;
            if (coord >= plane_val) has_high = 1;
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
// ==========================================================
// Checks that an SSBO bound to target has the expected size
// ==========================================================
void checkSSBOSize(GLenum target, GLsizeiptr expectedSize) {
    GLint size = 0;
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &size);
    if (size != expectedSize)
        printf("[ERROR] SSBO size mismatch: expected %ld, got %d\n", expectedSize, size);
    else
        printf("[DEBUG] SSBO size is correct (%d bytes).\n", size);
}


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
    // 1) Initialisation of glfw and creation of the off-screen window
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
    
    // ================================================
    // 2) Load mesh and solution files using libmeshb7
    // Reading the mesh file .mesh
    // ================================================
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
        tetsCPU[i].v[0] = (int)(vv[i] - 1);
        tetsCPU[i].v[1] = (int)(vv[i + Ntets] - 1);
        tetsCPU[i].v[2] = (int)(vv[i + 2*Ntets] - 1);
        tetsCPU[i].v[3] = (int)(vv[i + 3*Ntets] - 1);
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

    // ===========================================================
    // 3) Converting CPU -> GPU optimizing the memory and treament
    // ===========================================================
    t0 = millitime();
    struct FVertex { float x, y, z; };
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

    // ====================================
    // 4) Creation of the SSBO for the GPU
    // ====================================
    GLuint ssboVertices, ssboTets, ssboSol;
    // Generate an id for the buffer used for the vertices.
    glGenBuffers(1, &ssboVertices);
    // Bind the buffer to the target point GL_SHADER_STORAGE_BUFER, so that 
    // it can be used as a SSBO.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVertices);
    // Allocating and filling the buffer with the verticesGPU data
    // GL_STATIC_DRAW indicates the data won't change often
    // we want to read them mainly.
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nverts * sizeof(FVertex), verticesGPU, GL_STATIC_DRAW);
    // Bind our buffer to the specific binding point 0. 
    // Shaders will be able to access the data thanks to this.
    // This buffers allow the shaders to directly access the GPU memeory data.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVertices);

    // Same thing but for the tetrahedra.
    glGenBuffers(1, &ssboTets);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTets);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Ntets * sizeof(FTetrahedron), tetsGPU, GL_STATIC_DRAW);
    // Binding point 1
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboTets);
    
    // Same thing but for the solution
    glGenBuffers(1, &ssboSol);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSol);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Nsol * sizeof(float), solGPU, GL_STATIC_DRAW);
    
    // Binding point 2
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboSol);
    
    printf("##0\n");
    if(errno) printf("ERROR: %s", strerror(errno)), exit(1);


    // ======================================================================
    // Compute max absolute solution value for color normalization in shader.
    // Computing u_maxAbs
    // ======================================================================
    float u_maxAbs = 0.0f;
    {
        float minSol_val = solGPU[0], maxSol_val = solGPU[0];
        for (int i = 1; i < Nsol; i++){
            if(solGPU[i] < minSol_val) minSol_val = solGPU[i];
            if(solGPU[i] > maxSol_val) maxSol_val = solGPU[i];
        }
        u_maxAbs = fabs(minSol_val) > fabs(maxSol_val) ? fabs(minSol_val) : fabs(maxSol_val);
    }
    // INFO
    printf("[INFO] u_maxAbs = %f\n", u_maxAbs);
    
    // Verification
    printf("##1\n");
    if(errno) printf("[ERROR] location 1: %s", strerror(errno)), exit(1);

    // =============================================================
    // Create FBO and textures for off-screen rendering
    // =============================================================
    
    GLuint fbo, outputTex;
    // Generating a FBO, storing its ID, and binding it to GL_FRAME_BUFFER
    // So that following operation are applied to this FBO.
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    // Generating an Id for a texture, and binding it to GL_TEXTURE_2D
    // So that the following operations are applied to this texture.
    glGenTextures(1, &outputTex);
    glBindTexture(GL_TEXTURE_2D, outputTex);
    // Defining the texture picture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    // Defining the filtering parameters for the texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Binding th texture to the FBO
    // When the FBO is bound for rendering, all the color output (the drawn pixels) 
    // will be written into the outputTex texture via its attachment 
    // at GL_COLOR_ATTACHMENT0. 
    // Thus, the rendering is not sent directly to the screen, but to this 
    // texture, which is the basis for off-screen rendering.
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTex, 0);
    // Verification of the FBO
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[ERROR] Framebuffer not complete!\n");
        return EXIT_FAILURE;
    }
    // Detaching the FBO, we link the default framebuffer 0, once our FBO has
    // been set up. It does not suppress the FBO or its content. The texture
    // is still alright in the GPU. It is a way to return to the default FBO.
    // It prevents other rendering operations that could accidentally be executed
    // on our personalized FBO.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // OpenGL Verification
    printf("##2\n");
    if(errno) printf("ERROR location 2: %s", strerror(errno)), exit(1);

    // Creation of unique SSBOs for the grid in the loop (cells & indices)
    GLuint ssboGridCells, ssboGridIndices;
    glGenBuffers(1, &ssboGridCells);
    glGenBuffers(1, &ssboGridIndices);


    
    // OpenGL Verification
    printf("##3\n");
    if(errno) printf("ERROR location 3: %s", strerror(errno)), exit(1);

    // Création du compute shader pour le rendu off-screen
    GLuint computeProgram = createComputeProgram(computeShaderSource);
    glUseProgram(computeProgram);
    glUniform2i(glGetUniformLocation(computeProgram, "imageSize"), width, height);
    glUniform1i(glGetUniformLocation(computeProgram, "nCells_u"), nCells_u);
    glUniform1i(glGetUniformLocation(computeProgram, "nCells_v"), nCells_v);
    glUniform1f(glGetUniformLocation(computeProgram, "projMin1"), (float) proj_min1);
    glUniform1f(glGetUniformLocation(computeProgram, "projMax1"), (float) proj_max1);
    glUniform1f(glGetUniformLocation(computeProgram, "projMin2"), (float) proj_min2);
    glUniform1f(glGetUniformLocation(computeProgram, "projMax2"), (float) proj_max2);
    glUniform1f(glGetUniformLocation(computeProgram, "u_maxAbs"), u_maxAbs);
    int planeAxis = PLANE_X;
    glUniform1i(glGetUniformLocation(computeProgram, "planeAxis"), planeAxis);
    
    // Verification
    printf("##4\n");
    if(errno) printf("ERROR location 4: %s", strerror(errno)), exit(1);

    // ==================================================================
    // Picture creation loop via compute shader with timing measurements
    // ==================================================================
    int frameIndex = 0;
    double tTotal = millitime();
    int axis = PLANE_X;

    for (float pv = (float)xmin + 1.0f; pv <= (float)xmax; pv += 1.0f) {
        double tStart = millitime();
	printf("\n============= Processing for plane_val = %lf ==============\n\n", 
			pv);
	// ==================================================
	// ===== STEP A : FILTRATION OF THE TETRAHEDRA ======
	// ==================================================
	int *filtered_tets = NULL;
	int filtered_count = 0;
	if (!filter_tetrahedrons_parallel(Ntets, tetsCPU, verticesCPU, Nverts, axis, pv, &filtered_tets, &filtered_count)) {
		fprintf(stderr, "[ERROR] Failure of tetrahedra filtering for plane_val = %lf.\n", pv);
                return EXIT_FAILURE;
	} else {
		printf("[INFO] Number of filtered tetrahedra: %d of %" PRId64 "\n", filtered_count, Ntets);
	}
	printf("[INFO] Tet filtering took %.3f seconds\n", 0.001 * (millitime() - tStart));
        
	// ================================================= 
	// ======= STEP B : GRID INITIALIZATION =======
	// =================================================
	int totalCells = nCells_u * nCells_v;
	int *cellCounts = (int*) calloc(totalCells, sizeof(int));
	double cell_width  = (proj_max1 - proj_min1) / nCells_u;
	double cell_height = (proj_max2 - proj_min2) / nCells_v;

	// ======================================================================
	// ======= STEP C : INDEXATION OF FILTERED TETRAHEDRA IN THE GRID =======
	//                       BEGINNING GRID INDEXING 
	// ======================================================================
	// For each filtered tetrahedron
	for (int i = 0; i < filtered_count; i++) {
		int tet_idx = filtered_tets[i];
		double p_u_min = INFINITY, p_u_max = -INFINITY;
		double p_v_min = INFINITY, p_v_max = -INFINITY;
		// Projecting the 3D vertice in the 2D cut plan
		// We determine the position of the vertice in the 2D plan
		for (int j = 0; j < 4; j++) {
			double coord1, coord2;
			Vertex v = verticesCPU[tetsCPU[tet_idx].v[j]];
			// Choice of the projection with respect to the ax
			switch (axis) {
				case PLANE_X:
					coord1 = v.y; coord2 = v.z; break;
				case PLANE_Y:
					coord1 = v.x; coord2 = v.z; break;
				case PLANE_Z:
					coord1 = v.x; coord2 = v.y; break;
				default:
					coord1 = coord2 = 0;
			}
			// At the end of this loop, we get the smallest and biggest
			// value of u and v for this tetrahedron, which gives us its 
			// bounding box in the plan

			if (coord1 < p_u_min) p_u_min = coord1;
			if (coord1 > p_u_max) p_u_max = coord1;
			if (coord2 < p_v_min) p_v_min = coord2;
			if (coord2 > p_v_max) p_v_max = coord2;
		}

		// Determining the indexes of the cells our tetrahedron is covering
		// the first left(cell_u_min) and right cell(cell_u_max) and the 
		// first upper(cell_v_min) and lower cell(cell_v_max)
		int cell_u_min = (int)((p_u_min - proj_min1) / cell_width);
		int cell_u_max = (int)((p_u_max - proj_min1) / cell_width);
		int cell_v_min = (int)((p_v_min - proj_min2) / cell_height);
		int cell_v_max = (int)((p_v_max - proj_min2) / cell_height);
		
		// Verification
		if (cell_u_min < 0) cell_u_min = 0;
		if (cell_u_max >= nCells_u) cell_u_max = nCells_u - 1;
		if (cell_v_min < 0) cell_v_min = 0;
		if (cell_v_max >= nCells_v) cell_v_max = nCells_v - 1;
		for (int r = cell_u_min; r <= cell_u_max; r++) {
			for (int c = cell_v_min; c <= cell_v_max; c++) {
				int cellIndex = r * nCells_v + c;
				cellCounts[cellIndex]++;
			}
		}
	}
	// Allocation de la grille "aplatie"
	GridCell *gridCells = (GridCell*) malloc(totalCells * sizeof(GridCell));
	int totalIndices = 0;
	for (int i = 0; i < totalCells; i++) {
		gridCells[i].offset = totalIndices;
		gridCells[i].count = cellCounts[i];
		totalIndices += cellCounts[i];
	}
	free(cellCounts);

	// Allocating linear indexes arrays
	int *gridIndices = (int*) malloc(totalIndices * sizeof(int));
	// We fill this array by browsing the filtered tetrahedra again
	int *cellOffsets = (int*) calloc(totalCells, sizeof(int));
	for (int i = 0; i < filtered_count; i++) {
		int tet_idx = filtered_tets[i];
		double p_u_min = INFINITY, p_u_max = -INFINITY;
		double p_v_min = INFINITY, p_v_max = -INFINITY;
		for (int j = 0; j < 4; j++) {
			double coord1, coord2;
			Vertex v = verticesCPU[tetsCPU[tet_idx].v[j]];
			switch (axis) {
				case PLANE_X: 
					coord1 = v.y; coord2 = v.z; break;
				case PLANE_Y:
					coord1 = v.x; coord2 = v.z; break;
				case PLANE_Z:
					coord1 = v.x; coord2 = v.y; break;
				default:
					coord1 = coord2 = 0;
			}
			if (coord1 < p_u_min) p_u_min = coord1;
			if (coord1 > p_u_max) p_u_max = coord1;
			if (coord2 < p_v_min) p_v_min = coord2;
			if (coord2 > p_v_max) p_v_max = coord2;
		}
		int cell_u_min = (int)((p_u_min - proj_min1) / cell_width);
		int cell_u_max = (int)((p_u_max - proj_min1) / cell_width);
		int cell_v_min = (int)((p_v_min - proj_min2) / cell_height);
		int cell_v_max = (int)((p_v_max - proj_min2) / cell_height);
		if (cell_u_min < 0) cell_u_min = 0;
		if (cell_u_max >= nCells_u) cell_u_max = nCells_u - 1;
		if (cell_v_min < 0) cell_v_min = 0;
		if (cell_v_max >= nCells_v) cell_v_max = nCells_v - 1;
		for (int r = cell_u_min; r <= cell_u_max; r++) {
			for (int c = cell_v_min; c <= cell_v_max; c++) {
				int cell_index = r * nCells_v + c;
				int pos = gridCells[cell_index].offset + cellOffsets[cell_index];
				gridIndices[pos] = tet_idx;
				cellOffsets[cell_index]++;
			}
		}
	}
	free(cellOffsets);	    
	// INFO
	printf("[INFO] Total grid cells: %d\nTotal grid indices: %d\n", totalCells, totalIndices);
	printf("[INFO] Grid construction took %.3f seconds\n", 0.001*(millitime()-t0));

        // ==========================================
	// === END OF GRID CREATION & INDEXATION  ===
	// ==========================================

	// ===========================================
	// ====== Creation of SSBO for the grid ======
	// ===========================================
	// For the cells of the grid
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGridCells);
	glBufferData(GL_SHADER_STORAGE_BUFFER, totalCells * sizeof(GridCell), gridCells, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboGridCells);
	// For the indexes of the grid
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGridIndices);
	glBufferData(GL_SHADER_STORAGE_BUFFER, totalIndices * sizeof(int), gridIndices, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssboGridIndices);
	// Liberation once the data is send to the GPU buffers
	free(gridCells);
	free(gridIndices);

	// ==========================================
	// =========== SHADER & TEXTURES ============
	// ==========================================
        glUseProgram(computeProgram);
        glUniform1f(glGetUniformLocation(computeProgram, "planeVal"), pv);
        glUniform1i(glGetUniformLocation(computeProgram, "planeAxis"), planeAxis);
	// The compute shader writes its output to an image bound to the 
	// texture outputTex.
	// The first parameter (0) in glBindImageTexture binds outputTex to image unit 0,
	// which corresponds to the binding point declared in the compute shader (e.g.,
	// layout(rgba8, binding = 0) uniform writeonly image2D destImage;).
        glBindImageTexture(0, outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
	// glDispatchCompute(WORKGROUPS_X, WORKGROUPS_Y, WORKGROUPS_Z)
	// We create WORKGROUPS_X * WORKGROUPS_Y * WORKGROUPS_Z workgroups
	// each one of this work group contains local_size_x * local_size_y * local_size_z
	// invocations. 
	// By default, local_size_z = 1 when not specified.
	// So here we work with 1048576 invocations/threads, corresponding precisely to the        // number of pixels 1024*1024 we are working with.
	// Each thread execute the code of the compute shader.
	// This is where the compute shader is really launched.
        glDispatchCompute((GLuint)(width / 16), (GLuint)(height / 16), 1);
 	
	printf("##5\n");
	if(errno) printf("ERROR location 5: %s", strerror(errno)), exit(1);
  	

	// We bind the FBO once again to read the compute shader output
	// via glReadPixels in savePPM
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        
	printf("##6c\n");
	if(errno) printf("ERROR location 6c: %s", strerror(errno)), exit(1);
        
	// Set Viewport to cover the entire output image.
        glViewport(0, 0, width, height);
        
	printf("##6d\n");
	if(errno) printf("ERROR location 6d: %s", strerror(errno)), exit(1);
	
	// Create a filename based on the current frame index
	char filename[256];
        sprintf(filename, "movie_picture/computed_frame_%06d.ppm", frameIndex);
	
	printf("##6e\n");
	if(errno) printf("ERROR location 6e: %s", strerror(errno)), exit(1);
        
	// Save the rendered image to a PPM file (reads from the currently bound FBO)
        savePPM(filename, width, height);
	
	printf("##6f\n");
	if(errno) printf("ERROR location 6f: %s", strerror(errno)), exit(1);
        
	// Unbind the FBO (return to default framebuffer)
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	printf("##7\n");
	if(errno) printf("ERROR location 7: %s", strerror(errno)), exit(1);
        
	// Computing and printing the frame time
	double tFrame = millitime() - tStart;
	printf("[INFO] Processed plane and Saved image %s (planeVal = %.2f) in %.3f seconds\n", filename, pv, 0.001*tFrame);
        frameIndex++;
	//glFlush();
	free(filtered_tets);
    }

    tTotal = millitime() - tTotal;
    printf("[INFO] Total compute shader rendering time: %.3f seconds\n", 0.001*tTotal);

     glDeleteBuffers(1, &ssboVertices);
     glDeleteBuffers(1, &ssboTets);
     glDeleteBuffers(1, &ssboSol);
     glDeleteBuffers(1, &ssboGridCells);
     glDeleteBuffers(1, &ssboGridIndices);
     glDeleteProgram(computeProgram);
     glDeleteFramebuffers(1, &fbo);
     glDeleteTextures(1, &outputTex);

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



