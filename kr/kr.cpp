#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef unsigned char uchar;

struct uchar4 {
    uchar r;
    uchar g;
    uchar b;
    uchar a;
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
void print(vec3 v) {
    printf("%e %e %e\n", v.x, v.y, v.z);
}

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    uchar4 color;
};

trig trigs[38];

void build_space() {
    trigs[0] = {{-5, -5, 0}, {5, -5, 0}, {-5, 5, 0}, {100, 100, 100, 0}};
    trigs[1] = {{-5, 5, 0}, {5, -5, 0}, {5, 5, 0}, {100, 100, 100, 0}};

    trigs[2] = {{2.5, sqrt(3) / 2., 1}, {3, 0, 1}, {2, 0, 1}, {255, 0, 0, 0}};
    trigs[3] = {{2, 0, 1}, {3, 0, 1}, {2.5, sqrt(3) / 6., sqrt(2. / 3.) + 1}, {255, 0, 0, 0}};
    trigs[4] = {{3, 0, 1}, {2.5, sqrt(3) / 2., 1}, {2.5, sqrt(3) / 6., sqrt(2. / 3.) + 1}, {255, 0, 0, 0}};
    trigs[5] = {{2.5, sqrt(3) / 2., 1}, {2, 0, 1}, {2.5, sqrt(3) / 6., sqrt(2. / 3.) + 1}, {255, 0, 0, 0}};

    vec3 point_0 = {-3, 0, 1};
    vec3 point_1 = {-3, 1, 1};
    vec3 point_2 = {-2, 1, 1};
    vec3 point_3 = {-2, 0, 1};
    vec3 point_4 = {-3, 0, 2};
    vec3 point_5 = {-3, 1, 2};
    vec3 point_6 = {-2, 1, 2};
    vec3 point_7 = {-2, 0, 2};

    trigs[6] = {point_0, point_1, point_2, {0, 255, 0, 0}};
    trigs[7] = {point_2, point_3, point_0, {0, 255, 0, 0}};
    trigs[8] = {point_6, point_7, point_3, {0, 255, 0, 0}};
    trigs[9] = {point_3, point_2, point_6, {0, 255, 0, 0}};
    trigs[10] = {point_2, point_1, point_5, {0, 255, 0, 0}};
    trigs[11] = {point_5, point_6, point_2, {0, 255, 0, 0}};
    trigs[12] = {point_4, point_5, point_1, {0, 255, 0, 0}};
    trigs[13] = {point_1, point_0, point_4, {0, 255, 0, 0}};
    trigs[14] = {point_3, point_7, point_4, {0, 255, 0, 0}};
    trigs[15] = {point_4, point_0, point_3, {0, 255, 0, 0}};
    trigs[16] = {point_6, point_5, point_4, {0, 255, 0, 0}};
    trigs[17] = {point_4, point_7, point_6, {0, 255, 0, 0}};

    double phi = (1. + sqrt(5)) / 2.;
    point_0 = {0, 0.5, 2 * phi / 2. + 1};
    point_1 = {0, -0.5, 2 * phi / 2. + 1};
    point_2 = {0, 0.5, 1};
    point_3 = {0, -0.5, 1};
    point_4 = {0.5, phi / 2., phi / 2. + 1};
    point_5 = {-0.5, phi / 2., phi / 2. + 1};
    point_6 = {0.5, -phi / 2., phi / 2. + 1};   
    point_7 = {-0.5, -phi / 2., phi / 2. + 1};
    vec3 point_8 = {phi / 2., 0, 1.5 + phi / 2.};
    vec3 point_9 = {-phi / 2., 0, 1.5 + phi / 2.};
    vec3 point_10 = {phi / 2., 0, 0.5 + phi / 2.};
    vec3 point_11 = {-phi / 2., 0, 0.5 + phi / 2.};

    trigs[18] = {point_0, point_1, point_8, {0, 0, 255, 0}};
    trigs[19] = {point_0, point_8, point_4, {0, 0, 255, 0}};
    trigs[20] = {point_0, point_4, point_5, {0, 0, 255, 0}};
    trigs[21] = {point_0, point_5, point_9, {0, 0, 255, 0}};
    trigs[22] = {point_0, point_9, point_1, {0, 0, 255, 0}};

    trigs[23] = {point_2, point_3, point_11, {0, 0, 255, 0}};
    trigs[24] = {point_2, point_11, point_5, {0, 0, 255, 0}};
    trigs[25] = {point_2, point_5, point_4, {0, 0, 255, 0}};
    trigs[26] = {point_2, point_4, point_10, {0, 0, 255, 0}};
    trigs[27] = {point_2, point_10, point_3, {0, 0, 255, 0}};
    
    trigs[28] = {point_1, point_9, point_7, {0, 0, 255, 0}};
    trigs[29] = {point_1, point_7, point_6, {0, 0, 255, 0}};
    trigs[30] = {point_1, point_6, point_8, {0, 0, 255, 0}};

    trigs[31] = {point_3, point_10, point_6, {0, 0, 255, 0}};
    trigs[32] = {point_3, point_6, point_7, {0, 0, 255, 0}};
    trigs[33] = {point_3, point_7, point_11, {0, 0, 255, 0}};

    trigs[34] = {point_4, point_8, point_10, {0, 0, 255, 0}};
    trigs[35] = {point_5, point_11, point_9, {0, 0, 255, 0}};
    trigs[36] = {point_6, point_10, point_9, {0, 0, 255, 0}};
    trigs[37] = {point_7, point_9, point_11, {0, 0, 255, 0}};


    // for(int i = 0; i < 38; ++i) {
    //     print(trigs[i].a);
    //     print(trigs[i].b);
    //     print(trigs[i].c);
    //     print(trigs[i].a);
    //     printf("\n\n\n");
    // }
    // printf("\n\n\n");
}

int set_position(vec3 pos, vec3 dir, vec3 &pix_pos, vec3 &normal) {
    int k, k_min = -1;
    double ts_min;
    for (k = 0; k < 38; ++k) {
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
        if (ts < 0.) continue;
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
            pix_pos = add(pos, mult(dir, dir, dir, (vec3){ts, ts, ts}));
            normal = norm(prod(e1, e2));
        }
    }
    return k_min;
}

uchar4 ray(vec3 pos, vec3 dir, int count_lights, vec3 *lights) {
    vec3 pix_pos, normal;
    
    int k_min = set_position(pos, dir, pix_pos, normal);
    uchar4 pix_color;
    if (k_min == -1) pix_color = {0, 0, 0, 0};
    else pix_color = trigs[k_min].color;

    return pix_color;

    // // Базовый цвет треугольника (как float от 0 до 1)
    // double kd_r = pix_color.r / 255.0;
    // double kd_g = pix_color.g / 255.0;
    // double kd_b = pix_color.b / 255.0;
    // double ka_r = kd_r * 0.1; // ambient = 10% от диффузного
    // double ka_g = kd_g * 0.1;
    // double ka_b = kd_b * 0.1;
    // double ks_r = 1.0; // белый зеркальный блик
    // double ks_g = 1.0;
    // double ks_b = 1.0;
    // double shininess = 32.0;

    // double I_r = ka_r * 0.2;
    // double I_g = ka_g * 0.2;
    // double I_b = ka_b * 0.2;

    // vec3 view_dir = norm(diff(pos, pix_pos));

    // for (int i = 0; i < count_lights; ++i) {
    //     vec3 light_dir = norm(diff(lights[i], pix_pos));

    //     // // Пускаем луч от pix_pos в направлении light_dir
    //     // vec3 shadow_pos, shadow_normal;
    //     // int shadow_hit = set_position(lights[i], norm(diff(pix_pos, lights[i])), shadow_pos, shadow_normal);
    //     // if (shadow_hit != -1 && shadow_hit != k_min) return {0, 0, 0, 0};
    //     double NdotL = dot(normal, light_dir);
    //     if (NdotL < 0) NdotL = 0;

    //     I_r += kd_r * NdotL;
    //     I_g += kd_g * NdotL;
    //     I_b += kd_b * NdotL;
    // }

    // // Ограничение [0, 1]
    // if (I_r > 1.0) I_r = 1.0;
    // if (I_g > 1.0) I_g = 1.0;
    // if (I_b > 1.0) I_b = 1.0;
    // if (I_r < 0.0) I_r = 0.0;
    // if (I_g < 0.0) I_g = 0.0;
    // if (I_b < 0.0) I_b = 0.0;

    // return (uchar4){
    //     (uchar)(I_r * 255),
    //     (uchar)(I_g * 255),
    //     (uchar)(I_b * 255),
    //     0
    // };
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
            // printf("%d %d %d %d\n", data[(h - 1 - j) * w + i].r, data[(h - 1 - j) * w + i].g, data[(h - 1 - j) * w + i].b, data[(h - 1 - j) * w + i].a);
        }
    }
}

vec3 lights[1];

int main() {
    int w = 640, h = 480;
    int count_lights = 1;
    char buff[256];
    uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    vec3 *lights = (vec3*)malloc(sizeof(vec3) * count_lights);
    for (int i = 0; i < count_lights; ++i) {
        lights[i] = {0, 0, 5};
    }
    vec3 pc, pv;

    build_space();

    for (int k = 0; k < 100; ++k) {
        pc = (vec3) {6. * sin(0.05 * k), 6. * cos(0.05 * k), 5. + 2. * sin(0.1 * k)};
        pv = (vec3) {3. * sin(0.05 * k + M_PI), 3. * cos(0.5 * k + M_PI), 0.};
        render(pc, pv, w, h, 120, data, count_lights, lights);

        sprintf(buff, "res/%d.data", k);
        printf("%d: %s\n", k, buff);

        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }
    // for (int i = 0; i < w; ++i) {
    //     for (int j = 0; j < h; ++j) {
    //         printf("%d%d%d%d ", data[(h - 1 - j) * w + i].r, data[(h - 1 - j) * w + i].g, data[(h - 1 - j) * w + i].b, data[(h - 1 - j) * w + i].a);
    //     }
    //     printf("\n");
    // }
    free(data);
    return 0;
}