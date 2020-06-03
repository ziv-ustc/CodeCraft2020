/*
    团队 : 守望&静望&观望
    赛区 ：上合赛区
    成绩 ：0.1644 (初赛第五)
*/
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/wait.h>

using namespace std;
typedef unsigned int uint;

#define TEST 0
#define FREE 0
#define FREE_SHM 0
#define STORE_WAIT 0
#define ARM_NEON 1
#define DATA_SET 1004812 // 56 3738 38252 58284 77409 1004812 2755223 2861665(w) 2896262 3512444
#define MAX_DATA_SIZE 560000
#define MAX_OUTDEGREE_SIZE 50
#define MAX_INDEGREE_SIZE 50
#define MAX_RING_COUNT 2500000
#define MIN_DEPTH_LIMIT 3
#define MAX_DEPTH_LIMIT 7
#define LOAD_PROC_COUNT 4
#define PRETREAT_THREAD_COUNT 4
#define SEARCH_PROC_COUNT 4
#define STORE_PROC_COUNT 4
#define MAX_ID 50005

struct Fragment {
    Fragment(int u, int v): u(u), v(v) { }
    bool operator<(Fragment &rhs) {
        if (u != rhs.u) return u < rhs.u;
        return v < rhs.v;
    }
    int u;
    int v;
};

struct Ring {
    bool operator<(const Ring &rhs) {
        if (path[0] != rhs.path[0]) return path[0] < rhs.path[0];
        if (path[1] != rhs.path[1]) return path[1] < rhs.path[1];
        if (path[2] != rhs.path[2]) return path[2] < rhs.path[2];
        return true;
    }
    int path[3];
};

struct LoadInfo {
    int l;
    int r;
};

struct PretreatInfo {
    thread thd;
    int l;
    int r;
};

struct SearchInfo {
    size_t ranges[9];
    signed char *res_char[8];
    int  *res_char_cnt[8];
    vector<vector<Fragment>> PATH3;
    bool *record;
    int l;
    int r;
};

struct StoreInfo {
    int offset;
    int l[2];
    int r[2];
};

#if ARM_NEON
#include <arm_neon.h>
#define neon_int8x8_memcpy(dst,src) vst1_s8(dst, vld1_s8(src))
#endif

void load(const string &testFile, int &shmidInputs, uint* &inputs, int &inputSize, uint &idMax) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    const uint8_t ASCII[58] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9};
    int fd = open(testFile.c_str(), O_RDONLY);
    int data_size = lseek(fd, 0, SEEK_END);
    char *buf = (char*)mmap(NULL, data_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    shmidInputs = shmget(IPC_PRIVATE, LOAD_PROC_COUNT*(MAX_DATA_SIZE+2)*sizeof(uint), IPC_CREAT | 0600);
    LoadInfo load_infos[LOAD_PROC_COUNT];
    int cur = 0, batch = data_size / LOAD_PROC_COUNT;
    load_infos[0].l = cur;
    cur = batch - 1;
    while (buf[cur++] != '\n');
    load_infos[0].r = cur;
    load_infos[1].l = cur;
    cur = 2 * batch - 1;
    while (buf[cur++] != '\n');
    load_infos[1].r = cur;
    load_infos[2].l = cur;
    cur = 3 * batch - 1;
    while (buf[cur++] != '\n');
    load_infos[2].r = cur;
    load_infos[3].l = cur;
    cur = data_size - 1;
    while (buf[cur++] != '\n');
    load_infos[3].r = cur;
    int max_data_size_xload = LOAD_PROC_COUNT * MAX_DATA_SIZE;
    int max_data_size_xload_1 = max_data_size_xload + LOAD_PROC_COUNT;
    int proc_id;
    pid_t pid = 1;
    for (int pc = 1; pc < LOAD_PROC_COUNT; ++pc) {
        proc_id = pc;
        pid = fork();
        if (pid <= 0) break;
    }
    if (pid != 0) proc_id = 0;
    inputs = (uint*)shmat(shmidInputs, NULL, 0);
    uint id_max = 0, val = 0;
    int offset = proc_id * MAX_DATA_SIZE;
    int sz = offset;
    for (int i = load_infos[proc_id].l; i < load_infos[proc_id].r; ) {
        while (buf[i] != ',') val = val*10 + ASCII[buf[i++]];
        inputs[sz] = val;
        if (val > id_max) id_max = val;
        val = 0;
        ++sz;
        ++i;
        while (buf[i] != ',') val = val*10 + ASCII[buf[i++]];
        inputs[sz] = val;
        if (val > id_max) id_max = val;
        val = 0;
        ++sz;
        while (buf[i++] != '\n');
    }
    inputs[max_data_size_xload+proc_id] = sz-offset;
    inputs[max_data_size_xload_1+proc_id] = id_max;
#if TEST
    auto et1 = std::chrono::steady_clock::now();
    double ds1 = std::chrono::duration<double>(et1-st).count()*1000.0;
    cout << "the load time of process " << proc_id << " : " << ds1 << " ms." << endl;
#endif
    if (proc_id > 0) {
        shmdt(inputs);
        exit(0);
    }
    munmap(buf, data_size);
    int pc = 0;
    while (pc < 3) {
        pid = wait(NULL);
        if (pid > 0) ++pc;
    }
    inputSize = inputs[max_data_size_xload];
    idMax = MAX_ID;
    inputSize += inputs[max_data_size_xload+1];
    inputSize += inputs[max_data_size_xload+2];
    inputSize += inputs[max_data_size_xload+3];
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the load time : " << ds << " ms." << endl;
#endif
}

int (*outedges)[MAX_OUTDEGREE_SIZE];
int (*inedges)[MAX_INDEGREE_SIZE];
int *inedgeSize, *outedgeSize, *idLen;
int vtxCnt = 0, strMax;
signed char (*idComma)[11];
signed char (*idLF)[11];

void pretreat(int &shmidInputs, uint* &inputs, int &inputSize, uint &idMax) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    int sizeof_uint = sizeof(uint), sizeof_int = sizeof(int), sizeof_signed_char = sizeof(signed char);
    int *idtoidx = (int*)calloc(idMax+1, sizeof_int);
    uint *input = (uint*)malloc(inputSize*sizeof_uint);
    int offset = 0, max_data_size_xload = LOAD_PROC_COUNT * MAX_DATA_SIZE;
    memcpy(input+offset, inputs, inputs[max_data_size_xload]*sizeof_uint);
    offset += inputs[max_data_size_xload++];
    memcpy(input+offset, inputs+MAX_DATA_SIZE, inputs[max_data_size_xload]*sizeof_uint);
    offset += inputs[max_data_size_xload++];
    memcpy(input+offset, inputs+2*MAX_DATA_SIZE, inputs[max_data_size_xload]*sizeof_uint);
    offset += inputs[max_data_size_xload++];
    memcpy(input+offset, inputs+3*MAX_DATA_SIZE, inputs[max_data_size_xload]*sizeof_uint);
    offset += inputs[max_data_size_xload++];
    for (int i = 0; i < inputSize; i+=2) {
        if (input[i] > idMax || input[i+1] > idMax) continue;
        ++idtoidx[input[i]];
        ++idtoidx[input[i+1]];
    }
    for (int i = 0, idx = 1, i_end = idMax+1; i < i_end; ++i) {
        if (idtoidx[i] > 0) {
            idtoidx[i] = idx++;
            ++vtxCnt;
        }
    }
    uint *idxtoid = (uint*)malloc(vtxCnt*sizeof_uint);
    for (uint i = 0, j = 0, i_end = idMax+1; i < i_end; ++i) {
        if (idtoidx[i] > 0) {
            --idtoidx[i];
            idxtoid[j++] = i;
        }
    }
    outedges = (int(*)[MAX_OUTDEGREE_SIZE])malloc(vtxCnt*MAX_OUTDEGREE_SIZE*sizeof_int);
    inedges = (int(*)[MAX_INDEGREE_SIZE])malloc(vtxCnt*MAX_INDEGREE_SIZE*sizeof_int);
    outedgeSize = (int*)calloc(vtxCnt, sizeof_int);
    inedgeSize = (int*)calloc(vtxCnt, sizeof_int);
    int input_i, input_i_1;
    for (int i = 0; i < inputSize; i+=2) {
        if (input[i] > idMax || input[i+1] > idMax) continue;
        input_i = idtoidx[input[i]];
        input_i_1 = idtoidx[input[i+1]];
        outedges[input_i][outedgeSize[input_i]++] = input_i_1;
        inedges[input_i_1][inedgeSize[input_i_1]++] = input_i;
    }
    idComma = (signed char(*)[11])malloc(11*vtxCnt*sizeof_signed_char);
    idLF = (signed char(*)[11])malloc(11*vtxCnt*sizeof_signed_char);
    idLen = (int*)malloc(vtxCnt*sizeof_int);
    auto func = [&](PretreatInfo *infos) {
        for (int i = infos->l, i_end = infos->r, len_i = 0; i < i_end; ++i) {
            len_i = sprintf((char*)idComma[i], "%u", idxtoid[i]);
#if ARM_NEON
            neon_int8x8_memcpy(idLF[i], idComma[i]);
#else
            memcpy(idLF[i], idComma[i], len_i);
#endif
            idComma[i][len_i] = ',';
            idLF[i][len_i++] = '\n';
            idLen[i] = len_i;
        }
        for (int i = infos->l, i_end = infos->r; i < i_end; ++i) {
            sort(outedges[i], outedges[i]+outedgeSize[i]);
            sort(inedges[i], inedges[i]+inedgeSize[i]);
        }
    };
    PretreatInfo pret_infos[PRETREAT_THREAD_COUNT];
    int cur = 0, batch = vtxCnt / PRETREAT_THREAD_COUNT;
    pret_infos[0].l = cur;
    cur = batch;
    pret_infos[0].r = cur;
    pret_infos[1].l = cur;
    cur = 2 * batch;
    pret_infos[1].r = cur;
    pret_infos[2].l = cur;
    cur = 3 * batch;
    pret_infos[2].r = cur;
    pret_infos[3].l = cur;
    cur = vtxCnt;
    pret_infos[3].r = cur;
    pret_infos[0].thd = thread(func, &pret_infos[0]);
    pret_infos[1].thd = thread(func, &pret_infos[1]);
    pret_infos[2].thd = thread(func, &pret_infos[2]);
    pret_infos[3].thd = thread(func, &pret_infos[3]);
    pret_infos[0].thd.join();
    pret_infos[1].thd.join();
    pret_infos[2].thd.join();
    pret_infos[3].thd.join();
    strMax = idLen[vtxCnt-1];
#if FREE_SHM
    shmdt(inputs);
    shmctl(shmidInputs, IPC_RMID, NULL);
#endif
#if FREE
    free(idxtoid);
    free(idtoidx);
    free(input);
#endif
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the pretreat time : " << ds << " ms." << endl;
#endif
}

void dfs(SearchInfo *infos, int &head, int &cur, int *path) {
    infos->record[cur] = true;
    path[0] = cur;
    int res_char_cnt_0 = infos->res_char_cnt[0][0];
    int res_char_cnt_4 = infos->res_char_cnt[4][0];
    int res_char_cnt_5 = infos->res_char_cnt[5][0];
    int res_char_cnt_6 = infos->res_char_cnt[6][0];
    int res_char_cnt_7 = infos->res_char_cnt[7][0];
    int it_val, it2_val, it3_val, it4_val, u, v, sz;
    auto it_end = outedges[cur] + outedgeSize[cur];
    auto it = upper_bound(outedges[cur], it_end, head);
    for (; it != it_end; ++it) {
        it_val = *it;
        if(!infos->record[it_val]) {
            if(!infos->PATH3[it_val].empty()) {
                auto& midVtxs = infos->PATH3[it_val];
                sz = midVtxs.size();
                for (int i = 0; i < sz; ++i) {
                    u = midVtxs[i].u;
                    v = midVtxs[i].v;
                    if (!infos->record[u] && !infos->record[v]) {
#if ARM_NEON
                        neon_int8x8_memcpy(&infos->res_char[4][res_char_cnt_4], idComma[cur]);
                        res_char_cnt_4 += idLen[cur];
                        neon_int8x8_memcpy(&infos->res_char[4][res_char_cnt_4], idComma[it_val]);
                        res_char_cnt_4 += idLen[it_val];
                        neon_int8x8_memcpy(&infos->res_char[4][res_char_cnt_4], idComma[u]);
                        res_char_cnt_4 += idLen[u];
                        neon_int8x8_memcpy(&infos->res_char[4][res_char_cnt_4], idLF[v]);
                        res_char_cnt_4 += idLen[v];
                        ++res_char_cnt_0;
#else
                        memcpy(&infos->res_char[4][res_char_cnt_4], idComma[cur], idLen[cur]);
                        res_char_cnt_4 += idLen[cur];
                        memcpy(&infos->res_char[4][res_char_cnt_4], idComma[it_val], idLen[it_val]);
                        res_char_cnt_4 += idLen[it_val];
                        memcpy(&infos->res_char[4][res_char_cnt_4], idComma[u], idLen[u]);
                        res_char_cnt_4 += idLen[u];
                        memcpy(&infos->res_char[4][res_char_cnt_4], idLF[v], idLen[v]);
                        res_char_cnt_4 += idLen[v];
                        ++res_char_cnt_0;
#endif
                    }
                }
                infos->res_char_cnt[4][0] = res_char_cnt_4;
            }
            infos->record[it_val] = true;
            path[1] = it_val;
            auto it2_end = outedges[it_val] + outedgeSize[it_val];
            auto it2 = upper_bound(outedges[it_val], it2_end, head);
            for (; it2 != it2_end; ++it2) {
                it2_val = *it2;
                if(!infos->record[it2_val]) {
                    if(!infos->PATH3[it2_val].empty()) {
                        auto& midVtxs2 = infos->PATH3[it2_val];
                        sz = midVtxs2.size();
                        for (int i = 0; i < sz; ++i) {
                            u = midVtxs2[i].u;
                            v = midVtxs2[i].v;
                            if (!infos->record[u] && !infos->record[v]) {
#if ARM_NEON
                                neon_int8x8_memcpy(&infos->res_char[5][res_char_cnt_5], idComma[cur]);
                                res_char_cnt_5 += idLen[cur];
                                neon_int8x8_memcpy(&infos->res_char[5][res_char_cnt_5], idComma[it_val]);
                                res_char_cnt_5 += idLen[it_val];
                                neon_int8x8_memcpy(&infos->res_char[5][res_char_cnt_5], idComma[it2_val]);
                                res_char_cnt_5 += idLen[it2_val];
                                neon_int8x8_memcpy(&infos->res_char[5][res_char_cnt_5], idComma[u]);
                                res_char_cnt_5 += idLen[u];
                                neon_int8x8_memcpy(&infos->res_char[5][res_char_cnt_5], idLF[v]);
                                res_char_cnt_5 += idLen[v];
                                ++res_char_cnt_0;
#else
                                memcpy(&infos->res_char[5][res_char_cnt_5], idComma[cur], idLen[cur]);
                                res_char_cnt_5 += idLen[cur];
                                memcpy(&infos->res_char[5][res_char_cnt_5], idComma[it_val], idLen[it_val]);
                                res_char_cnt_5 += idLen[it_val];
                                memcpy(&infos->res_char[5][res_char_cnt_5], idComma[it2_val], idLen[it2_val]);
                                res_char_cnt_5 += idLen[it2_val];
                                memcpy(&infos->res_char[5][res_char_cnt_5], idComma[u], idLen[u]);
                                res_char_cnt_5 += idLen[u];
                                memcpy(&infos->res_char[5][res_char_cnt_5], idLF[v], idLen[v]);
                                res_char_cnt_5 += idLen[v];
                                ++res_char_cnt_0;
#endif
                            }
                        }
                        infos->res_char_cnt[5][0] = res_char_cnt_5;
                    }
                    infos->record[it2_val] = true;
                    path[2] = it2_val;
                    auto it3_end = outedges[it2_val] + outedgeSize[it2_val];
                    auto it3 = upper_bound(outedges[it2_val], it3_end, head);
                    for (; it3 != it3_end; ++it3) {
                        it3_val = *it3;
                        if(!infos->record[it3_val]) {
                            if(!infos->PATH3[it3_val].empty()) {
                                auto& midVtxs3 = infos->PATH3[it3_val];
                                sz = midVtxs3.size();
                                for (int i = 0; i < sz; ++i) {
                                    u = midVtxs3[i].u;
                                    v = midVtxs3[i].v;
                                    if (!infos->record[u] && !infos->record[v]) {
#if ARM_NEON
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idComma[cur]);
                                        res_char_cnt_6 += idLen[cur];
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it_val]);
                                        res_char_cnt_6 += idLen[it_val];
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it2_val]);
                                        res_char_cnt_6 += idLen[it2_val];
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it3_val]);
                                        res_char_cnt_6 += idLen[it3_val];
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idComma[u]);
                                        res_char_cnt_6 += idLen[u];
                                        neon_int8x8_memcpy(&infos->res_char[6][res_char_cnt_6], idLF[v]);
                                        res_char_cnt_6 += idLen[v];
                                        ++res_char_cnt_0;
#else
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idComma[cur], idLen[cur]);
                                        res_char_cnt_6 += idLen[cur];
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it_val], idLen[it_val]);
                                        res_char_cnt_6 += idLen[it_val];
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it2_val], idLen[it2_val]);
                                        res_char_cnt_6 += idLen[it2_val];
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idComma[it3_val], idLen[it3_val]);
                                        res_char_cnt_6 += idLen[it3_val];
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idComma[u], idLen[u]);
                                        res_char_cnt_6 += idLen[u];
                                        memcpy(&infos->res_char[6][res_char_cnt_6], idLF[v], idLen[v]);
                                        res_char_cnt_6 += idLen[v];
                                        ++res_char_cnt_0;
#endif
                                    }
                                }
                                infos->res_char_cnt[6][0] = res_char_cnt_6;
                            }
                            infos->record[it3_val] = true;
                            path[3] = it3_val;
                            auto it4_end = outedges[it3_val] + outedgeSize[it3_val];
                            auto it4 = upper_bound(outedges[it3_val], it4_end, head);
                            for (; it4 != it4_end; ++it4) {
                                it4_val = *it4;
                                if(!infos->record[it4_val]) {
                                    if(!infos->PATH3[it4_val].empty()) {
                                        auto& midVtxs4 = infos->PATH3[it4_val];
                                        sz = midVtxs4.size();
                                        for (int i = 0; i < sz; ++i) {
                                            u = midVtxs4[i].u;
                                            v = midVtxs4[i].v;
                                            if (!infos->record[u] && !infos->record[v]) {
#if ARM_NEON
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[cur]);
                                                res_char_cnt_7 += idLen[cur];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it_val]);
                                                res_char_cnt_7 += idLen[it_val];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it2_val]);
                                                res_char_cnt_7 += idLen[it2_val];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it3_val]);
                                                res_char_cnt_7 += idLen[it3_val];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it4_val]);
                                                res_char_cnt_7 += idLen[it4_val];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idComma[u]);
                                                res_char_cnt_7 += idLen[u];
                                                neon_int8x8_memcpy(&infos->res_char[7][res_char_cnt_7], idLF[v]);
                                                res_char_cnt_7 += idLen[v];
                                                ++res_char_cnt_0;
#else
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[cur], idLen[cur]);
                                                res_char_cnt_7 += idLen[cur];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it_val], idLen[it_val]);
                                                res_char_cnt_7 += idLen[it_val];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it2_val], idLen[it2_val]);
                                                res_char_cnt_7 += idLen[it2_val];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it3_val], idLen[it3_val]);
                                                res_char_cnt_7 += idLen[it3_val];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[it4_val], idLen[it4_val]);
                                                res_char_cnt_7 += idLen[it4_val];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idComma[u], idLen[u]);
                                                res_char_cnt_7 += idLen[u];
                                                memcpy(&infos->res_char[7][res_char_cnt_7], idLF[v], idLen[v]);
                                                res_char_cnt_7 += idLen[v];
                                                ++res_char_cnt_0;
#endif
                                            }
                                        }
                                        infos->res_char_cnt[7][0] = res_char_cnt_7;
                                    }
                                }
                            }
                            infos->record[it3_val] = false;
                        }
                    }
                    infos->record[it2_val] = false;
                }
            }
            infos->record[it_val] = false;
        }
    }
    infos->record[cur] = false;
    infos->res_char_cnt[0][0] = res_char_cnt_0;
}

SearchInfo search_infos[SEARCH_PROC_COUNT];

void search(int &shmidResChar, int &shmidResCharCnt) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    float vtx_cof[SEARCH_PROC_COUNT] = {0.0463, 0.1284, 0.2342, 1.0};
    for (int pc = 0, last = 0; pc < SEARCH_PROC_COUNT; ++pc) {
        search_infos[pc].l = last;
        int cur = vtx_cof[pc] * vtxCnt;
        search_infos[pc].r = cur;
        last = cur;
    }
    const int char_len_max = MAX_RING_COUNT * strMax;
    float ring_char_max[8] = {0.0, 0.0, 0.0, 0.3*char_len_max, 0.4*char_len_max, 0.5*char_len_max, 1.8*char_len_max, 6.3*char_len_max};
    size_t last = 0, cur = 0;
    for (int i = MIN_DEPTH_LIMIT; i <= MAX_DEPTH_LIMIT; ++i) {
        for (int pc = 0; pc < SEARCH_PROC_COUNT; ++pc) {
            search_infos[pc].ranges[i] = last;
            cur = last + int(ring_char_max[i]);
            search_infos[pc].ranges[i+1] = cur;
            last = cur;
        }
    }
    shmidResChar = shmget(IPC_PRIVATE, search_infos[SEARCH_PROC_COUNT-1].ranges[8]*sizeof(signed char), IPC_CREAT | 0600);
    shmidResCharCnt = shmget(IPC_PRIVATE, 6*SEARCH_PROC_COUNT*sizeof(int), IPC_CREAT | 0600);
    int proc_id;
    pid_t pid = 1;
    for (int pc = 1; pc < SEARCH_PROC_COUNT; ++pc) {
        proc_id = pc;
        pid = fork();
        if (pid <= 0) break;
    }
    if (pid != 0) proc_id = 0;
    Ring *res3 = (Ring*)malloc(MAX_RING_COUNT*0.1*sizeof(Ring));
    int res3_cnt = 0;
    signed char *res_char_ptr = (signed char*)shmat(shmidResChar, NULL, 0);
    int *res_char_cnt_ptr = (int*)shmat(shmidResCharCnt, NULL, 0);
    search_infos[proc_id].res_char[3] = res_char_ptr + search_infos[proc_id].ranges[3];
    search_infos[proc_id].res_char_cnt[3] = res_char_cnt_ptr + proc_id;
    search_infos[proc_id].res_char_cnt[3][0] = 0;
    search_infos[proc_id].res_char[4] = res_char_ptr + search_infos[proc_id].ranges[4];
    search_infos[proc_id].res_char_cnt[4] = res_char_cnt_ptr + proc_id + SEARCH_PROC_COUNT;
    search_infos[proc_id].res_char_cnt[4][0] = 0;
    search_infos[proc_id].res_char[5] = res_char_ptr + search_infos[proc_id].ranges[5];
    search_infos[proc_id].res_char_cnt[5] = res_char_cnt_ptr + proc_id + 2*SEARCH_PROC_COUNT;
    search_infos[proc_id].res_char_cnt[5][0] = 0;
    search_infos[proc_id].res_char[6] = res_char_ptr + search_infos[proc_id].ranges[6];
    search_infos[proc_id].res_char_cnt[6] = res_char_cnt_ptr + proc_id + 3*SEARCH_PROC_COUNT;
    search_infos[proc_id].res_char_cnt[6][0] = 0;
    search_infos[proc_id].res_char[7] = res_char_ptr + search_infos[proc_id].ranges[7];
    search_infos[proc_id].res_char_cnt[7] = res_char_cnt_ptr + proc_id + 4*SEARCH_PROC_COUNT;
    search_infos[proc_id].res_char_cnt[7][0] = 0;
    search_infos[proc_id].res_char_cnt[0] = res_char_cnt_ptr + proc_id + 5*SEARCH_PROC_COUNT;
    search_infos[proc_id].res_char_cnt[0][0] = 0;
    search_infos[proc_id].record = (bool*)calloc(vtxCnt, sizeof(bool));
    search_infos[proc_id].PATH3 = vector<vector<Fragment>>(vtxCnt);
    int path[7];
    const int vtx_num = vtxCnt;
    int rec[vtx_num];
    int rec_sz = 0;
    int outedge_sz, inedge_sz, itj_val, itk_val, itm_val;
    for (int i = search_infos[proc_id].l, i_end = search_infos[proc_id].r; i < i_end; ++i) {
        outedge_sz = outedgeSize[i];
        inedge_sz = inedgeSize[i];
        if (outedge_sz && outedges[i][outedge_sz-1] > i && inedge_sz && inedges[i][inedge_sz-1] > i) {
            auto itj_end = inedges[i] + inedge_sz;
            auto itj = upper_bound(inedges[i], itj_end, i);
            for (; itj != itj_end; ++itj) {
                itj_val = *itj;
                auto itk_end = inedges[itj_val] + inedgeSize[itj_val];
                auto itk = upper_bound(inedges[itj_val], itk_end, i);
                for (; itk != itk_end; ++itk) {
                    itk_val = *itk;
                    auto itm_end = inedges[itk_val] + inedgeSize[itk_val];
                    auto itm = lower_bound(inedges[itk_val], itm_end, i);
                    for (; itm != itm_end; ++itm) {
                        itm_val = *itm;
                        if (itm_val == itj_val) continue;
                        if (itm_val != i) {
                            search_infos[proc_id].PATH3[itm_val].emplace_back(Fragment(itk_val,itj_val));
                            if (search_infos[proc_id].PATH3[itm_val].size() == 1) rec[rec_sz++] = itm_val;
                        } else {
                            res3[res3_cnt].path[0] = i;
                            res3[res3_cnt].path[1] = itk_val;
                            res3[res3_cnt++].path[2] = itj_val;
                            ++search_infos[proc_id].res_char_cnt[0][0];
                        }
                    }
                }
            }
            if (rec_sz == 0) continue;
            for (int i = 0; i < rec_sz; ++i) sort(search_infos[proc_id].PATH3[rec[i]].begin(), search_infos[proc_id].PATH3[rec[i]].end());
            dfs(&search_infos[proc_id], i, i, path);
            for (int i = 0; i < rec_sz; ++i) search_infos[proc_id].PATH3[rec[i]].clear();
            rec_sz = 0;
        }
    }
    sort(res3, res3+res3_cnt);
    int res_char_cnt = search_infos[proc_id].res_char_cnt[3][0], res_path_k;
    for (int i = 0; i < res3_cnt; ++i) {
#if ARM_NEON
        res_path_k = res3[i].path[0];
        neon_int8x8_memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idComma[res_path_k]);
        res_char_cnt += idLen[res_path_k];
        res_path_k = res3[i].path[1];
        neon_int8x8_memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idComma[res_path_k]);
        res_char_cnt += idLen[res_path_k];
        res_path_k = res3[i].path[2];
        neon_int8x8_memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idLF[res_path_k]);
        res_char_cnt += idLen[res_path_k];
#else
        res_path_k = res3[i].path[0];
        memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idComma[res_path_k], idLen[res_path_k]);
        res_char_cnt += idLen[res_path_k];
        res_path_k = res3[i].path[1];
        memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idComma[res_path_k], idLen[res_path_k]);
        res_char_cnt += idLen[res_path_k];
        res_path_k = res3[i].path[2];
        memcpy(&search_infos[proc_id].res_char[3][res_char_cnt], idLF[res_path_k], idLen[res_path_k]);
        res_char_cnt += idLen[res_path_k];
#endif
    }
    search_infos[proc_id].res_char_cnt[3][0] = res_char_cnt;
#if FREE
    free(res3);
    free(search_infos[proc_id].record);
#endif
#if TEST
    auto et1 = std::chrono::steady_clock::now();
    double ds1 = std::chrono::duration<double>(et1-st).count()*1000.0;
    cout << "the search time of process " << proc_id << " : " << ds1 << " ms." << endl;
#endif
    if (proc_id > 0) {
        shmdt(res_char_ptr);
        shmdt(res_char_cnt_ptr);
        exit(0);
    }
    int pc = 0;
    while (pc < SEARCH_PROC_COUNT-1) {
        pid = wait(NULL);
        if (pid > 0) ++pc;
    }
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the search time : " << ds << " ms." << endl;
#endif
}

void store(const string &resultFile, const int &shmidResChar, const int &shmidResCharCnt) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    signed char *res_char_ptr = (signed char*)shmat(shmidResChar, NULL, 0);
    int *res_char_cnt_ptr = (int*)shmat(shmidResCharCnt, NULL, 0);
    const int search_proc_cnt_x5 = SEARCH_PROC_COUNT * 5;
    int res_cnt = 0;
    for (int pc = 0; pc < SEARCH_PROC_COUNT; ++pc) {
        res_cnt += *(res_char_cnt_ptr + search_proc_cnt_x5 + pc);
    }
    signed char *res_char[search_proc_cnt_x5];
    int res_char_cnt[search_proc_cnt_x5];
    int res_char_sum[search_proc_cnt_x5+1];
    memset(res_char_sum, 0, sizeof(res_char_sum));
    const int search_proc_count_xi[8] = {0, 0, 0, 0, SEARCH_PROC_COUNT, 2*SEARCH_PROC_COUNT, 3*SEARCH_PROC_COUNT, 4*SEARCH_PROC_COUNT};
    for (int i = MIN_DEPTH_LIMIT, id = 0; i <= MAX_DEPTH_LIMIT; ++i) {
        for (int pc = 0; pc < SEARCH_PROC_COUNT; ++pc) {
            res_char_cnt[id] = *(res_char_cnt_ptr + search_proc_count_xi[i] + pc);
            res_char_sum[id+1] = res_char_sum[id] + res_char_cnt[id];
            res_char[id++] = res_char_ptr + search_infos[pc].ranges[i];
        }
    }
    int res_char_total = res_char_sum[search_proc_cnt_x5];
    string res_cnt_str = to_string(res_cnt) + '\n';
    int res_cnt_str_sz = res_cnt_str.size();
    int fd = open(resultFile.c_str(), O_RDWR | O_CREAT, 00777);
    lseek(fd, res_cnt_str_sz+res_char_total-1, SEEK_SET); 
    write(fd, " ", 1);
    signed char *buf = (signed char*)mmap(NULL, res_cnt_str_sz+res_char_total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    signed char *buf_base = buf;
    close(fd);
    memcpy(buf_base, res_cnt_str.data(), res_cnt_str_sz);
    buf_base += res_cnt_str_sz;
    StoreInfo store_infos[STORE_PROC_COUNT];
    int last[2] = {0, 0};
    int offset_tmp = 0;
    for (int pc = 0, i = 0; pc < STORE_PROC_COUNT; ++pc) {
        store_infos[pc].offset = offset_tmp;
        offset_tmp = (pc+1) * res_char_total / STORE_PROC_COUNT;
        for (; i < search_proc_cnt_x5; ++i) {
            if (offset_tmp <= res_char_sum[i+1]) {
                store_infos[pc].l[0] = last[0];
                store_infos[pc].l[1] = last[1];
                store_infos[pc].r[0] = i;
                store_infos[pc].r[1] = offset_tmp - res_char_sum[i];
                break;
            }
        }
        last[0] = store_infos[pc].r[0];
        last[1] = store_infos[pc].r[1];
    }
    int proc_id;
    pid_t pid = 1;
    for (int pc = 1; pc < STORE_PROC_COUNT; ++pc) {
        proc_id = pc;
        pid = fork();
        if (pid <= 0) break;
    }
    if (pid != 0) proc_id = 0;
    int len;
    signed char *buffer = buf_base + store_infos[proc_id].offset;
    int l_0 = store_infos[proc_id].l[0], l_1 = store_infos[proc_id].l[1];
    int r_0 = store_infos[proc_id].r[0], r_1 = store_infos[proc_id].r[1];
    for (int i = l_0; i <= r_0; ++i) {
        if (l_0 == r_0) {
            len = r_1 - l_1;
            memcpy(buffer, res_char[i]+l_1, len);
            buffer += len;
        } else {
            if (i == l_0) {
                len = res_char_cnt[i] - l_1;
                memcpy(buffer, res_char[i]+l_1, len);
                buffer += len;
            } else if (i == r_0) {
                len = r_1;
                memcpy(buffer, res_char[i], len);
                buffer += len;
            } else {
                memcpy(buffer, res_char[i], res_char_cnt[i]);
                buffer += res_char_cnt[i];
            }
        }
    }
#if TEST
        auto et1 = std::chrono::steady_clock::now();
        double ds1 = std::chrono::duration<double>(et1-st).count()*1000.0;
        cout << "the store time of process " << proc_id << " : " << ds1 << " ms." << endl;
#endif
    if (proc_id > 0) exit(0);
#if STORE_WAIT
    int pc = 0;
    while (pc < STORE_PROC_COUNT-1) {
        pid = wait(NULL);
        if (pid > 0) ++pc;
    }
    munmap(buf, res_cnt_str_sz+res_char_total);
#if FREE_SHM
    shmdt(res_char_ptr);
    shmdt(res_char_cnt_ptr);
    shmctl(shmidResChar, IPC_RMID,NULL);
    shmctl(shmidResCharCnt, IPC_RMID,NULL);
#endif
#if FREE
    free(outedges);
    free(inedges);
    free(outedgeSize);
    free(inedgeSize);
    free(idLen);
#endif
#endif
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "The store time : " << ds << " ms." << endl;
#endif
}

int main() {
    string testFile = "/data/test_data.txt";
    string resultFile = "/projects/student/result.txt";
#if TEST
    testFile = string("../../data/") + to_string(DATA_SET) + string("/test_data.txt");
    resultFile = string("../../data/") + to_string(DATA_SET) + string("/result.txt");
    auto st = std::chrono::steady_clock::now();
#endif
    int shmidInputs, inputSize;
    uint *inputs, idMax;
    load(testFile, shmidInputs, inputs, inputSize, idMax);
    pretreat(shmidInputs, inputs, inputSize, idMax);
    int shmidResChar, shmidResCharCnt;
    search(shmidResChar, shmidResCharCnt);
    store(resultFile, shmidResChar, shmidResCharCnt);
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the elapsed time : " << ds << " ms." << endl;
#endif
    exit(0);
}
