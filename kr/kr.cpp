#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef unsigned char uchar;

struct uchar4 {
    uchar r = 0;
    uchar g = 0;
    uchar b = 0;
    uchar a = 0;
};

struct vec3 {
    double x;
    double y;
    double z;
};

double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 prod(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    if (l == 0) return {0, 0, 0};
    return {v.x / l, v.y / l, v.z / l};
}

vec3 diff(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 add(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return {
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z,
    };
}

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    uchar4 color;
    double k_refl = 0.5;
    double k_refr = 0.0;
    bool has_texture = false;
};

trig trigs[38];

// Глобальные переменные текстуры
uchar* texture_data = nullptr;
int tex_width = 0, tex_height = 0, tex_channels = 0;

void build_space(vec3 tetr_c, vec3 hex_c, vec3 iko_c, double tetr_r, double hex_r, double iko_r,
                 double tetr_k_refl, double tetr_k_refr,
                 double hex_k_refl, double hex_k_refr,
                 double iko_k_refl, double iko_k_refr) {
    // Пол — два треугольника
    trigs[0] = {{-5, -5, 0}, {5, -5, 0}, {-5, 5, 0}, {100, 100, 100, 0}, 0.0, 0.0, true};
    trigs[1] = {{-5, 5, 0}, {5, -5, 0}, {5, 5, 0}, {100, 100, 100, 0}, 0.0, 0.0, true};

    vec3 point_0 = {tetr_r * (-1.632990) + tetr_c.x, tetr_r * (-0.942809) + tetr_c.y, tetr_r * (-0.666667) + tetr_c.z};
    vec3 point_1 = {tetr_r * 0. + tetr_c.x, tetr_r * 1.885620 + tetr_c.y, tetr_r * (-0.666667) + tetr_c.z};
    vec3 point_2 = {tetr_r * 1.632990 + tetr_c.x, tetr_r * (-0.942809) + tetr_c.y, tetr_r * (-0.666667) + tetr_c.z};
    vec3 point_3 = {tetr_r * 0. + tetr_c.x, tetr_r * 0. + tetr_c.y, tetr_r * 2. + tetr_c.z};

    trigs[2] = {point_0, point_1, point_2, {255, 0, 0, 0}, tetr_k_refl, tetr_k_refr};
    trigs[3] = {point_0, point_3, point_1, {255, 0, 0, 0}, tetr_k_refl, tetr_k_refr};
    trigs[4] = {point_1, point_3, point_2, {255, 0, 0, 0}, tetr_k_refl, tetr_k_refr};
    trigs[5] = {point_2, point_3, point_0, {255, 0, 0, 0}, tetr_k_refl, tetr_k_refr};

    vec3 p0 = {hex_c.x - 0.5 * hex_r, hex_c.y - 0.5 * hex_r, hex_c.z - 0.5 * hex_r};
    vec3 p1 = {hex_c.x - 0.5 * hex_r, hex_c.y + 0.5 * hex_r, hex_c.z - 0.5 * hex_r};
    vec3 p2 = {hex_c.x + 0.5 * hex_r, hex_c.y + 0.5 * hex_r, hex_c.z - 0.5 * hex_r};
    vec3 p3 = {hex_c.x + 0.5 * hex_r, hex_c.y - 0.5 * hex_r, hex_c.z - 0.5 * hex_r};
    vec3 p4 = {hex_c.x - 0.5 * hex_r, hex_c.y - 0.5 * hex_r, hex_c.z + 0.5 * hex_r};
    vec3 p5 = {hex_c.x - 0.5 * hex_r, hex_c.y + 0.5 * hex_r, hex_c.z + 0.5 * hex_r};
    vec3 p6 = {hex_c.x + 0.5 * hex_r, hex_c.y + 0.5 * hex_r, hex_c.z + 0.5 * hex_r};
    vec3 p7 = {hex_c.x + 0.5 * hex_r, hex_c.y - 0.5 * hex_r, hex_c.z + 0.5 * hex_r};

    trigs[6]  = {p0, p1, p2, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[7]  = {p0, p2, p3, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[8]  = {p4, p7, p6, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[9]  = {p4, p6, p5, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[10] = {p4, p5, p1, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[11] = {p4, p1, p0, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[12] = {p3, p2, p6, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[13] = {p3, p6, p7, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[14] = {p0, p3, p7, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[15] = {p0, p7, p4, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[16] = {p1, p5, p6, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};
    trigs[17] = {p1, p6, p2, {0, 255, 0, 0}, hex_k_refl, hex_k_refr};

    vec3 point_4 = {-0.850651 * iko_r + iko_c.x, 0. * iko_r + iko_c.y, 0.525731 * iko_r + iko_c.z};
    vec3 point_5 = {-0.525731 * iko_r + iko_c.x, 0.850651 * iko_r + iko_c.y, 0. * iko_r + iko_c.z};
    vec3 point_6 = {0.525731 * iko_r + iko_c.x, 0.850651 * iko_r + iko_c.y, 0. * iko_r + iko_c.z};
    vec3 point_7 = {0.525731 * iko_r + iko_c.x, -0.850651 * iko_r + iko_c.y, 0. * iko_r + iko_c.z};
    vec3 point_8 = {-0.525731 * iko_r + iko_c.x, -0.850651 * iko_r + iko_c.y, 0. * iko_r + iko_c.z};
    vec3 point_9 = {0. * iko_r + iko_c.x, -0.525731 * iko_r + iko_c.y, -0.850651 * iko_r + iko_c.z};
    vec3 point_10 = {0. * iko_r + iko_c.x, 0.525731 * iko_r + iko_c.y, -0.850651 * iko_r + iko_c.z};
    vec3 point_11 = {0. * iko_r + iko_c.x, 0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z};

    trigs[18] = {{0.850651 * iko_r + iko_c.x, 0., 0.525731 * iko_r + iko_c.z},
                 {0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_6, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[19] = {{0.850651 * iko_r + iko_c.x, 0., 0.525731 * iko_r + iko_c.z},
                 point_7,
                 {0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[20] = {{-0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_4, point_5, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[21] = {point_4,
                 {-0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_8, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[22] = {point_6, point_5, point_11, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[23] = {point_5, point_6, point_10, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[24] = {point_9, point_10,
                 {0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[25] = {point_10, point_9,
                 {-0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[26] = {point_7, point_8, point_9, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[27] = {point_8, point_7,
                 {0., -0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[28] = {point_11,
                 {0., -0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z},
                 {0.850651 * iko_r + iko_c.x, 0., 0.525731 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[29] = {{0., -0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z},
                 point_11, point_4, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[30] = {point_6,
                 {0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_10, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[31] = {{0.850651 * iko_r + iko_c.x, 0., 0.525731 * iko_r + iko_c.z},
                 point_6, point_11, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[32] = {{-0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_5, point_10, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[33] = {point_5, point_4, point_11, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[34] = {{0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_7, point_9, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[35] = {point_7,
                 {0.850651 * iko_r + iko_c.x, 0., 0.525731 * iko_r + iko_c.z},
                 {0., -0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[36] = {{-0.850651 * iko_r + iko_c.x, 0., -0.525731 * iko_r + iko_c.z},
                 point_9, point_8, {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
    trigs[37] = {point_4, point_8,
                 {0., -0.525731 * iko_r + iko_c.y, 0.850651 * iko_r + iko_c.z},
                 {0, 0, 255, 0}, iko_k_refl, iko_k_refr};
}

void set_position(vec3 pos, vec3 dir, vec3 &pix_pos, vec3 &normal, int &k_min, double &ts_min) {
    k_min = -1;
    ts_min = 1e300;

    for (int k = 0; k < 38; ++k) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) continue;

        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0. || u > 1.) continue;

        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0. || u + v > 1.) continue;

        double ts = dot(q, e2) / div;
        if (ts < 1e-6) continue;

        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
            pix_pos = add(pos, mult(dir, dir, dir, (vec3){ts, ts, ts}));
            normal = norm(prod(e1, e2));
            if (dot(dir, normal) > 0) {
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }
        }
    }
}

vec3 reflect(vec3 I, vec3 N) {
    double dotIN = dot(I, N);
    return (vec3){
        I.x - 2.0 * dotIN * N.x,
        I.y - 2.0 * dotIN * N.y,
        I.z - 2.0 * dotIN * N.z
    };
}

uchar4 shade(vec3 point, vec3 normal, uchar4 base_color, int count_lights, vec3 *lights, bool has_texture) {
    uchar4 tex_color = base_color;
    if (has_texture && texture_data) {
        // UV: от -5 до +5 → [0, 10] → [0,1]
        double uu = (point.x + 5.0) / 10.0;
        double vv = (point.y + 5.0) / 10.0;

        // Wrap (повторение текстуры)
        uu = fmod(uu, 1.0);
        vv = fmod(vv, 1.0);
        if (uu < 0) uu += 1.0;
        if (vv < 0) vv += 1.0;

        int x = (int)(uu * tex_width) % tex_width;
        int y = (int)(vv * tex_height) % tex_height;
        if (x < 0) x += tex_width;
        if (y < 0) y += tex_height;

        uchar* pixel = &texture_data[(y * tex_width + x) * 3];
        tex_color = { pixel[0], pixel[1], pixel[2], 0 };
    }

    double kd_r = tex_color.r / 255.0;
    double kd_g = tex_color.g / 255.0;
    double kd_b = tex_color.b / 255.0;

    // Ambient
    double I_r = kd_r * 0.1;
    double I_g = kd_g * 0.1;
    double I_b = kd_b * 0.1;

    // Diffuse
    for (int i = 0; i < count_lights; ++i) {
        vec3 light_dir = norm(diff(lights[i], point));
        double NdotL = dot(normal, light_dir);
        if (NdotL < 0) NdotL = 0;
        I_r += kd_r * NdotL;
        I_g += kd_g * NdotL;
        I_b += kd_b * NdotL;
    }

    I_r = fmax(0.0, fmin(1.0, I_r));
    I_g = fmax(0.0, fmin(1.0, I_g));
    I_b = fmax(0.0, fmin(1.0, I_b));

    return (uchar4){
        (uchar)(I_r * 255),
        (uchar)(I_g * 255),
        (uchar)(I_b * 255),
        0
    };
}

uchar4 ray(vec3 pos, vec3 dir, int count_lights, vec3 *lights) {
    vec3 pix_pos, normal;
    int k_min;
    double ts;
    set_position(pos, dir, pix_pos, normal, k_min, ts);

    if (k_min == -1) {
        return (uchar4){0, 0, 0, 0};
    }

    uchar4 base_lit = shade(pix_pos, normal, trigs[k_min].color, count_lights, lights, trigs[k_min].has_texture);
    double I_r = base_lit.r / 255.0;
    double I_g = base_lit.g / 255.0;
    double I_b = base_lit.b / 255.0;

    double ks = trigs[k_min].k_refl;
    if (ks > 0.0) {
        vec3 refl_dir = reflect(dir, normal);
        double eps = 1e-5;
        vec3 offset_pos = add(pix_pos, mult(normal, normal, normal, (vec3){eps, eps, eps}));

        vec3 refl_target, refl_normal;
        int refl_k;
        double refl_ts;
        set_position(offset_pos, refl_dir, refl_target, refl_normal, refl_k, refl_ts);

        if (refl_k != -1) {
            uchar4 refl_lit = shade(refl_target, refl_normal, trigs[refl_k].color, count_lights, lights, trigs[refl_k].has_texture);
            double refl_r = refl_lit.r / 255.0;
            double refl_g = refl_lit.g / 255.0;
            double refl_b = refl_lit.b / 255.0;

            I_r = (1.0 - ks) * I_r + ks * refl_r;
            I_g = (1.0 - ks) * I_g + ks * refl_g;
            I_b = (1.0 - ks) * I_b + ks * refl_b;

            I_r = fmax(0.0, fmin(1.0, I_r));
            I_g = fmax(0.0, fmin(1.0, I_g));
            I_b = fmax(0.0, fmin(1.0, I_b));
        }
    }

    return (uchar4){
        (uchar)(I_r * 255),
        (uchar)(I_g * 255),
        (uchar)(I_b * 255),
        0
    };
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, int count_lights, vec3 *lights) {
    double dw = 2. / (w - 1);
    double dh = 2. / (h - 1);
    double z = 1. / tan(angle * M_PI / 360.);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0., 0., 1.}));
    vec3 by = prod(bx, bz);

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            vec3 v = {-1. + dw * i, (-1. + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, dir, count_lights, lights);
        }
    }
}

int main() {
    // Загрузка текстуры
    texture_data = stbi_load("floor.jpg", &tex_width, &tex_height, &tex_channels, 3);
    if (!texture_data) {
        fprintf(stderr, "ERROR: Cannot load floor.jpg\n");
        return 1;
    }

    int frames;
    scanf("%d", &frames);

    char buff[256];
    char buff2[256];
    scanf("%s", buff);

    int w, h, angle;
    scanf("%d %d %d", &w, &h, &angle);
    uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);

    double r_0_c, z_0_c, phi_0_c, a_r_c, a_z_c, w_r_c, w_z_c, w_phi_c, p_r_c, p_z_c;
    double r_0_n, z_0_n, phi_0_n, a_r_n, a_z_n, w_r_n, w_z_n, w_phi_n, p_r_n, p_z_n;
    scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &r_0_c, &z_0_c, &phi_0_c, &a_r_c, &a_z_c, &w_r_c, &w_z_c, &w_phi_c, &p_r_c, &p_z_c);
    scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &r_0_n, &z_0_n, &phi_0_n, &a_r_n, &a_z_n, &w_r_n, &w_z_n, &w_phi_n, &p_r_n, &p_z_n);

    vec3 tetr_c, hex_c, iko_c;
    double tetr_r, hex_r, iko_r;
    double tetr_k_refl, tetr_k_refr, hex_k_refl, hex_k_refr, iko_k_refl, iko_k_refr;
    scanf("%lf %lf %lf %lf %lf %lf", &tetr_c.x, &tetr_c.y, &tetr_c.z, &tetr_r, &tetr_k_refl, &tetr_k_refr);
    scanf("%lf %lf %lf %lf %lf %lf", &hex_c.x, &hex_c.y, &hex_c.z, &hex_r, &hex_k_refl, &hex_k_refr);
    scanf("%lf %lf %lf %lf %lf %lf", &iko_c.x, &iko_c.y, &iko_c.z, &iko_r, &iko_k_refl, &iko_k_refr);

    int count_lights;
    scanf("%d", &count_lights);
    
    vec3 *lights = (vec3*)malloc(sizeof(vec3) * count_lights);
    for (int i = 0; i < count_lights; ++i) {
        scanf("%lf %lf %lf", &lights[i].x, &lights[i].y, &lights[i].z);
    }
    
    vec3 pc, pv;

    build_space(tetr_c, hex_c, iko_c, tetr_r, hex_r, iko_r,
                tetr_k_refl, tetr_k_refr,
                hex_k_refl, hex_k_refr,
                iko_k_refl, iko_k_refr);

    for (int k = 0; k < frames; ++k) {
        double r_c = r_0_c + a_r_c * sin(w_r_c * k + p_r_c);
        double phi_c = phi_0_c + w_phi_c * k;
        pc = (vec3) {r_c * cos(phi_c), r_c * sin(phi_c), z_0_c + a_z_c * sin(w_z_c * k + p_z_c)};
        
        double r_n = r_0_n + a_r_n * sin(w_r_n * k + p_r_n);
        double phi_n = phi_0_n + w_phi_n * k;
        pv = (vec3) {r_n * cos(phi_n), r_n * sin(phi_n), z_0_n + a_z_n * sin(w_z_n * k + p_z_n)};
        render(pc, pv, w, h, angle, data, count_lights, lights);

        sprintf(buff2, buff, k);
        printf("%d: %s\n", k, buff2);

        FILE *out = fopen(buff2, "wb");
        if (out) {
            fwrite(&w, sizeof(int), 1, out);
            fwrite(&h, sizeof(int), 1, out);
            fwrite(data, sizeof(uchar4), w * h, out);
            fclose(out);
        }
    }

    free(data);
    free(lights);
    stbi_image_free(texture_data);
    return 0;
}