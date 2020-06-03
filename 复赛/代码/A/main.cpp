/*
    团队 : 守望&静望&观望
    赛区 ：上合赛区
    成绩 ：2.7704 (复赛A榜第五）
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

#define TEST 0
#define FREE 1
#define FREE_SHM 1
#define STORE_WAIT 0
#define ARM_NEON 1
// 18 43 56 3738 38252 58284 77409 1004812 2755223 2861665(w) 2896262 3512444
// sparse: 19077789 19469882 19875681 20214386 dense: 11148929 18934561 19859553 20811056 21495805 open: 19630345
#define DATA_SET 19630345
#define MAX_EDGE_COUNT 2000000  // 6000000 2048*3000
#define MAX_RING_COUNT 4096*125 // 20000000
#define MAX_RING_CHAR_COUNT_3 3*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_4 4*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_5 5*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_6 6*11*MAX_RING_COUNT*7 // 0.7
#define MAX_RING_CHAR_COUNT_7 7*11*MAX_RING_COUNT*9 // 0.9
#define PATH3_SIZE 100000
#define MAX_VTX_COUNT 2000000
#define MIN_DEPTH_LIMIT 3
#define MAX_DEPTH_LIMIT 7
#define LOAD_PROC_COUNT 4
#define PRETREAT_THREAD_COUNT 4
#define SEARCH_THREAD_COUNT 4
#define STORE_PROC_COUNT 5

#define Y_MIN 429496729 // INT_MAX/5
#define X_MIN 715827882 // INT_MAX/3

struct Fragment {
    int w;
    int u;
    int v;
    int f;
    Fragment(): w(0), u(0), v(0), f(0) { }
    Fragment(int &w, int &u, int &v, int &f): w(w), u(u), v(v), f(f) { }
    bool operator<(Fragment &rhs) {
        if (w != rhs.w) return w < rhs.w;
        if (u != rhs.u) return u < rhs.u;
        return v < rhs.v;
    }
};

struct Transfer {
    int idx;
    int cash;
    Transfer(): idx(0), cash(0) { }
    Transfer(int i, int c): idx(i), cash(c) { }
    bool operator<(const Transfer &rhs) const {
        return idx < rhs.idx;
    }
};

struct Ring {
    int path[3];
    bool operator<(const Ring &rhs) {
        if (path[0] != rhs.path[0]) return path[0] < rhs.path[0];
        if (path[1] != rhs.path[1]) return path[1] < rhs.path[1];
        if (path[2] != rhs.path[2]) return path[2] < rhs.path[2];
        return true;
    }
};

struct Edge {
    int id1;
    int id2;
    int cash;
    bool isInedge;
    bool operator<(const Edge &rhs) const {
        return id1 < rhs.id1;
    }
};

struct LoadInfo {
    int l;
    int r;
};

struct PretreatInfo {
    int l;
    int r;
};

struct StoreInfo {
    size_t offset;
    int beg;
    int end;
    char th[5];
    char len;
};

#if ARM_NEON
#include <arm_neon.h>
inline void neon_int8x8_memcpy(volatile signed char *dst,  volatile signed char *src) {
    asm volatile  (
                // "prfm pldl1keep, [%[src], #64] \n"
                "ld1 {v0.16b},[%[src]] \n"
                "st1 {v0.16b},[%[dst]] \n"
                : [dst]"+r"(dst)
                : [src]"r"(src)
                : "v0","cc","memory");
}
#endif

#define check_cash(X,X_3,Y,Y_5) ((X > X_MIN || Y <= X_3) && (Y > Y_MIN || X <= Y_5))

void load(const string &testFile, int &shmidInputs, Edge* &inputs, int &inputSize, int &idMax) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    const uint8_t ASCII[58] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9};
    int fd = open(testFile.c_str(), O_RDONLY);
    int data_char_size = lseek(fd, 0, SEEK_END);
    char *buf = (char*)mmap(NULL, data_char_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    LoadInfo load_infos[LOAD_PROC_COUNT];
    for (int pc = 0, last = 0, cur = 0; pc < LOAD_PROC_COUNT; ++pc) {
        load_infos[pc].l = last;
        cur = (pc+1) * data_char_size / LOAD_PROC_COUNT - 1;
        while (buf[cur++] != '\n');
        load_infos[pc].r = cur;
        last = cur;
    }
    int max_data_size_xload = LOAD_PROC_COUNT * MAX_EDGE_COUNT * 2;
    int max_data_size_xload_1 = max_data_size_xload + LOAD_PROC_COUNT;
    shmidInputs = shmget(IPC_PRIVATE, (max_data_size_xload_1+LOAD_PROC_COUNT)*sizeof(Edge), IPC_CREAT | 0600);
    int proc_id;
    pid_t pid = 1;
    for (int pc = 1; pc < LOAD_PROC_COUNT; ++pc) {
        proc_id = pc;
        pid = fork();
        if (pid <= 0) break;
    }
    if (pid != 0) proc_id = 0;
    inputs = (Edge*)shmat(shmidInputs, NULL, 0);
    int id_max = 0, val = 0;
    int offset = proc_id * MAX_EDGE_COUNT * 2;
    int sz1 = offset, sz2;
    for (int i = load_infos[proc_id].l, i_end = load_infos[proc_id].r; i < i_end; ) {
        while (buf[i] != ',') val = (val<<3) + (val<<1) + ASCII[buf[i++]];
        sz2 = sz1+1;
        inputs[sz1].id1 = val;
        inputs[sz1].isInedge = false;
        inputs[sz2].id2 = val;
        if (val > id_max) id_max = val;
        val = 0;
        ++i;
        while (buf[i] != ',') val = (val<<3) + (val<<1) + ASCII[buf[i++]];
        inputs[sz1].id2 = val;
        inputs[sz2].id1 = val;
        inputs[sz2].isInedge = true;
        if (val > id_max) id_max = val;
        val = 0;
        while (buf[i] != '\r' && buf[i] != '\n') val = (val<<3) + (val<<1) + ASCII[buf[i++]];
        inputs[sz1].cash = val;
        inputs[sz2].cash = val;
        sz1 += 2;
        val = 0;
        while (buf[i++] != '\n');
    }
    inputs[max_data_size_xload+proc_id].id1 = sz1-offset;
    sort(inputs+offset, inputs+sz1);
    inputs[max_data_size_xload_1+proc_id].id1 = id_max;
#if TEST
    auto et1 = std::chrono::steady_clock::now();
    double ds1 = std::chrono::duration<double>(et1-st).count()*1000.0;
    cout << "the load time of process " << proc_id << " : " << ds1 << " ms." << endl;
#endif
    if (proc_id > 0) {
        shmdt(inputs);
        exit(0);
    }
    munmap(buf, data_char_size);
    int pc = 0;
    while (pc < 3) {
        pid = wait(NULL);
        if (pid > 0) ++pc;
    }
    inputSize = 0;
    for (int pc = 0; pc < LOAD_PROC_COUNT; ++pc) {
        inputSize += inputs[max_data_size_xload+pc].id1;
        if (idMax < inputs[max_data_size_xload_1+pc].id1) idMax = inputs[max_data_size_xload_1+pc].id1;
    }
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the load time : " << ds << " ms." << endl;
#endif
}

int vtxCnt;
pair<int,int> *shared_pool;
pair<int,int> *outedges[MAX_VTX_COUNT], *inedges[MAX_VTX_COUNT];
int outedgeSize[MAX_VTX_COUNT], inedgeSize[MAX_VTX_COUNT];
signed char idComma[MAX_VTX_COUNT][16];
signed char idLF[MAX_VTX_COUNT][16];
int idLen[MAX_VTX_COUNT];

void pretreat(int &shmidInputs, Edge* &inputs, int &inputSize, int &idMax) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    int sizeof_int = sizeof(int), sizeof_intptr = sizeof(int*), sizeof_pair = sizeof(pair<int,int>), sizeof_edge = sizeof(Edge);
    int sizeof_intx64 = sizeof_int*64;
    int max_data_size_xload = LOAD_PROC_COUNT * MAX_EDGE_COUNT * 2;
    Edge *input_ptr[LOAD_PROC_COUNT];
    int input_sz[LOAD_PROC_COUNT], offset[LOAD_PROC_COUNT] = {0};
    for (int pc = 0, last = 0; pc < LOAD_PROC_COUNT; ++pc) {
        input_ptr[pc] = inputs + pc * MAX_EDGE_COUNT * 2;
        input_sz[pc] = inputs[max_data_size_xload+pc].id1;
        offset[pc] = last;
        last = offset[pc] + input_sz[pc];
    }
    Edge *input_tmp = (Edge*)malloc(inputSize*sizeof_edge);
    Edge *input = (Edge*)malloc(inputSize*sizeof_edge);
    auto merge_func = [&](const int &id) {
        int id_x2 = id * 2;
        int id_x2_1 = id_x2 + 1;
        merge(input_ptr[id_x2], input_ptr[id_x2]+input_sz[id_x2], input_ptr[id_x2_1], input_ptr[id_x2_1]+input_sz[id_x2_1], input_tmp+offset[id_x2]);
    };
    thread thd0(merge_func, 0);
    thread thd1(merge_func, 1);
    thd0.join();
    thd1.join();
    merge(input_tmp, input_tmp+offset[2], input_tmp+offset[2], input_tmp+inputSize, input);
    char id_low;
    int id, id_high, vtx_cnt = 0;
    int bitmap_sz = (idMax>>6) + 1;
    int **idtoidx = (int**)calloc(bitmap_sz, sizeof_intptr);
    int *input_beg = (int*)malloc((inputSize+1)*sizeof_int);
    for (int i = 0, j = 0; i < inputSize; i = j) {
        id = input[i].id1;
        id_high = id >> 6;
        id_low = id & 63;
        input_beg[vtx_cnt] = i;
        if (idtoidx[id_high] == NULL) idtoidx[id_high] = (int*)malloc(sizeof_intx64);
        idtoidx[id_high][id_low] = vtx_cnt;
        while (j < inputSize && input[j].id1 == id) {
            if (input[j].isInedge) ++inedgeSize[vtx_cnt];
            else ++outedgeSize[vtx_cnt];
            ++j;
        }
        ++vtx_cnt;
    }
    input_beg[vtx_cnt] = inputSize;
    int *idxtoid = (int*)malloc(vtx_cnt*sizeof_int);
    shared_pool = (pair<int,int>*)malloc(inputSize*sizeof_pair);
    size_t pool_offset[4] = {0};
    {
        size_t num1, num2, num3;
        const int vtx_cnt1 = vtx_cnt/4, vtx_cnt2 = vtx_cnt/2, vtx_cnt3 = 3*vtx_cnt/4;
        auto addfunc = [](const int &beg, const int &end, size_t &num) {
            num = 0;
            for (int i = beg; i < end; ++i) {
                num += outedgeSize[i];
                num += inedgeSize[i];
            }
        };
        thread th0(addfunc,0,vtx_cnt1,ref(num1));
        thread th1(addfunc,vtx_cnt1,vtx_cnt2,ref(num2));
        thread th2(addfunc,vtx_cnt2,vtx_cnt3,ref(num3));
        th0.join();
        th1.join();
        th2.join();
        pool_offset[1] = num1;
        pool_offset[2] = num1 + num2;;
        pool_offset[3] = pool_offset[2] + num3;
    }
    auto create_func = [&](const int &beg, const int& end, int id) {
        int sz1, sz2;
        int input_beg_;
        int offset = pool_offset[id];
        for (int i = beg, j = beg; i < end; ++i) {
            input_beg_ = input_beg[i];
            id = input[input_beg_].id1;
            idxtoid[i] = id;
            auto &outedges_idx = outedges[i];
            auto &inedges_idx = inedges[i];
            outedges_idx = shared_pool + offset;
            offset += outedgeSize[i];
            inedges_idx = shared_pool + offset;
            offset += inedgeSize[i];
            sz1 = 0;
            sz2 = 0;
            j = input_beg_;
            while (j < input_beg[i+1]) {
                if (input[j].isInedge) {
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    inedges_idx[sz1].first = idtoidx[high][low];
                    inedges_idx[sz1++].second = int(input[j++].cash);
                } else {
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    outedges_idx[sz2].first = idtoidx[high][low];
                    outedges_idx[sz2++].second = int(input[j++].cash);
                }
            }
        }
    };
    {
        const int vtx_cnt1 = vtx_cnt/4, vtx_cnt2 = vtx_cnt/2, vtx_cnt3 = 3*vtx_cnt/4;
        thread th0(create_func,0,vtx_cnt>>2,0);
        thread th1(create_func,vtx_cnt>>2,vtx_cnt>>1,1);
        thread th2(create_func,vtx_cnt>>1,3*vtx_cnt>>2,2);
        thread th3(create_func,3*vtx_cnt>>2,vtx_cnt,3);
        th0.join();
        th1.join();
        th2.join();
        th3.join();
    }
    auto transfer_func = [&](PretreatInfo *infos) {
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
            sort(outedges[i], outedges[i]+outedgeSize[i], [](pair<int,int>&a, pair<int,int>&b)->bool{return a.first < b.first;});
            sort(inedges[i], inedges[i]+inedgeSize[i], [](pair<int,int>&a, pair<int,int>&b)->bool{return a.first < b.first;});
        }
    };
    PretreatInfo pret_infos[PRETREAT_THREAD_COUNT];
    for (int tc = 0, last = 0, cur = 0; tc < PRETREAT_THREAD_COUNT; ++tc) {
        pret_infos[tc].l = last;
        cur = (tc+1) * vtx_cnt / PRETREAT_THREAD_COUNT;
        pret_infos[tc].r = cur;
        last = cur;
    }
    thread pret_thd[PRETREAT_THREAD_COUNT];
    for (int tc = 0; tc < PRETREAT_THREAD_COUNT; ++tc) pret_thd[tc] = thread(transfer_func, &pret_infos[tc]);
    for (int tc = 0; tc < PRETREAT_THREAD_COUNT; ++tc) pret_thd[tc].join();
    vtxCnt = vtx_cnt;
#if FREE_SHM
    shmdt(inputs);
    shmctl(shmidInputs, IPC_RMID, NULL);
#endif
#if FREE
    for (int i = 0; i < vtxCnt; ++i) {
        id_high = idxtoid[i] >> 6;
        if (idtoidx[id_high] != NULL) {
            free(idtoidx[id_high]);
            idtoidx[id_high] = NULL;
        }
    }
    free(input_beg);
    free(idxtoid);
    free(idtoidx);
    free(input);
    free(input_tmp);
#endif
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the pretreat time : " << ds << " ms." << endl;
#endif
}

//SearchInfo search_infos[SEARCH_THREAD_COUNT];
signed char res3_char[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_3];
signed char res4_char[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_4];
signed char res5_char[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_5];
signed char res6_char[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_6];
signed char res7_char[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_7];
signed char *res_char[SEARCH_THREAD_COUNT][8];
size_t res_char_cnt[SEARCH_THREAD_COUNT][8] = {0};
signed char *vtx_char[5][MAX_VTX_COUNT];
size_t vtx_char_cnt[5][SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
Ring res3[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_3/33];
int res_cnt[SEARCH_THREAD_COUNT] = {0};
Fragment PATH3[SEARCH_THREAD_COUNT][PATH3_SIZE];
int PATH3_INFO[SEARCH_THREAD_COUNT][MAX_VTX_COUNT][2];
bool record[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
bool is_PATH3_front[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
bool is_PATH3_back[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
atomic<int> vtx_cnt_atomic(0);

void dfs(int head, pair<int,int>* it1, int tid) {
    size_t res_char_cnt_4 = res_char_cnt[tid][4], res_char_cnt_5 = res_char_cnt[tid][5];
    size_t res_char_cnt_6 = res_char_cnt[tid][6], res_char_cnt_7 = res_char_cnt[tid][7];
    int len_head = idLen[head], len_idx1, len_idx2, len_idx3, len_idx4;
    int res_cnt_total = res_cnt[tid];
    int cash1_x3 = (it1->second<<1)+it1->second;
    len_idx1 = idLen[it1->first];
    if (is_PATH3_front[tid][it1->first]) {
        for (int i = PATH3_INFO[tid][it1->first][0], end = PATH3_INFO[tid][it1->first][1]+PATH3_INFO[tid][it1->first][0]; i < end; ++i) {
            if (!is_PATH3_back[tid][PATH3[tid][i].v]) continue;
            int u = PATH3[tid][i].u;
            int v = PATH3[tid][i].v;
            int f = PATH3[tid][i].f;
            int f_x5 = (f<<2)+f;
            if (check_cash(it1->second, cash1_x3, f, f_x5)) {
#if ARM_NEON
                neon_int8x8_memcpy(&res_char[tid][4][res_char_cnt_4], idComma[head]);
                res_char_cnt_4 += len_head;
                neon_int8x8_memcpy(&res_char[tid][4][res_char_cnt_4], idComma[it1->first]);
                res_char_cnt_4 += len_idx1;
                neon_int8x8_memcpy(&res_char[tid][4][res_char_cnt_4], idComma[u]);
                res_char_cnt_4 += idLen[u];
                neon_int8x8_memcpy(&res_char[tid][4][res_char_cnt_4], idLF[v]);
                res_char_cnt_4 += idLen[v];
                ++res_cnt_total;
#else
                memcpy(&res_char[tid][4][res_char_cnt_4], idComma[head], len_head);
                res_char_cnt_4 += len_head;
                memcpy(&res_char[tid][4][res_char_cnt_4], idComma[it1->first], len_idx1);
                res_char_cnt_4 += len_idx1;
                memcpy(&res_char[tid][4][res_char_cnt_4], idComma[u], idLen[u]);
                res_char_cnt_4 += idLen[u];
                memcpy(&res_char[tid][4][res_char_cnt_4], idLF[v], idLen[v]);
                res_char_cnt_4 += idLen[v];
                ++res_cnt_total;
#endif
            }
        }
    }
    record[tid][it1->first] = true;
    auto it2_end = outedges[it1->first]+outedgeSize[it1->first];
    for (auto it2 = outedges[it1->first]; it2 != it2_end; ++it2) {
        if (it2->first <= head) continue;
        int cash2_x3 = (it2->second<<1)+it2->second;
        int cash2_x5 = (it2->second<<2)+it2->second;
        len_idx2 = idLen[it2->first];
        if (check_cash(it1->second, cash1_x3, it2->second, cash2_x5)) {
            if (is_PATH3_front[tid][it2->first]) {
                for (int i = PATH3_INFO[tid][it2->first][0], end = PATH3_INFO[tid][it2->first][1]+PATH3_INFO[tid][it2->first][0]; i < end; ++i) {
                    if (!is_PATH3_back[tid][PATH3[tid][i].v] || record[tid][PATH3[tid][i].u] || record[tid][PATH3[tid][i].v]) continue;
                    int u = PATH3[tid][i].u;
                    int v = PATH3[tid][i].v;
                    int f = PATH3[tid][i].f;
                    int f_x5 = (f<<2)+f;
                    if (check_cash(it2->second, cash2_x3, f, f_x5)) {
#if ARM_NEON
                        neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[head]);
                        res_char_cnt_5 += len_head;
                        neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[it1->first]);
                        res_char_cnt_5 += len_idx1;
                        neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[it2->first]);
                        res_char_cnt_5 += len_idx2;
                        neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[u]);
                        res_char_cnt_5 += idLen[u];
                        neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idLF[v]);
                        res_char_cnt_5 += idLen[v];
                        ++res_cnt_total;
#else
                        memcpy(&res_char[tid][5][res_char_cnt_5], idComma[head], len_head);
                        res_char_cnt_5 += len_head;
                        memcpy(&res_char[tid][5][res_char_cnt_5], idComma[it1->first], len_idx1);
                        res_char_cnt_5 += len_idx1;
                        memcpy(&res_char[tid][5][res_char_cnt_5], idComma[it2->first], len_idx2);
                        res_char_cnt_5 += len_idx2;
                        memcpy(&res_char[tid][5][res_char_cnt_5], idComma[u], idLen[u]);
                        res_char_cnt_5 += idLen[u];
                        memcpy(&res_char[tid][5][res_char_cnt_5], idLF[v], idLen[v]);
                        res_char_cnt_5 += idLen[v];
                        ++res_cnt_total;
#endif
                    }
                }
            }
            record[tid][it2->first] = true;
            auto it3_end = outedges[it2->first]+outedgeSize[it2->first];
            for (auto it3 = outedges[it2->first]; it3 != it3_end; ++it3) {
                if (it3->first <= head || it3->first == it1->first) continue;
                int cash3_x3 = (it3->second<<1)+it3->second;
                int cash3_x5 = (it3->second<<2)+it3->second;
                len_idx3 = idLen[it3->first];
                if (check_cash(it2->second, cash2_x3, it3->second, cash3_x5)) {
                    if (is_PATH3_front[tid][it3->first]) {
                        for (int i = PATH3_INFO[tid][it3->first][0], end = PATH3_INFO[tid][it3->first][1]+PATH3_INFO[tid][it3->first][0]; i < end; ++i) {
                            if (!is_PATH3_back[tid][PATH3[tid][i].v] || record[tid][PATH3[tid][i].v] || record[tid][PATH3[tid][i].u]) continue;
                            int u = PATH3[tid][i].u;
                            int v = PATH3[tid][i].v;
                            int f = PATH3[tid][i].f;
                            int f_x5 = (f<<2)+f;
                            if (check_cash(it3->second, cash3_x3, f, f_x5)) {
#if ARM_NEON
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[head]);
                                res_char_cnt_6 += len_head;
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it1->first]);
                                res_char_cnt_6 += len_idx1;
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it2->first]);
                                res_char_cnt_6 += len_idx2;
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it3->first]);
                                res_char_cnt_6 += len_idx3;
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[u]);
                                res_char_cnt_6 += idLen[u];
                                neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idLF[v]);
                                res_char_cnt_6 += idLen[v];
                                ++res_cnt_total;
#else
                                memcpy(&res_char[tid][6][res_char_cnt_6], idComma[head], len_head);
                                res_char_cnt_6 += len_head;
                                memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it1->first], len_idx1);
                                res_char_cnt_6 += len_idx1;
                                memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it2->first], len_idx2);
                                res_char_cnt_6 += len_idx2;
                                memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it3->first], len_idx3);
                                res_char_cnt_6 += len_idx3;
                                memcpy(&res_char[tid][6][res_char_cnt_6], idComma[u], idLen[u]);
                                res_char_cnt_6 += idLen[u];
                                memcpy(&res_char[tid][6][res_char_cnt_6], idLF[v], idLen[v]);
                                res_char_cnt_6 += idLen[v];
                                ++res_cnt_total;
#endif
                            }
                        }
                    }
                    record[tid][it3->first] = true;
                    auto it4_end = outedges[it3->first]+outedgeSize[it3->first];
                    for (auto it4 = outedges[it3->first]; it4 != it4_end; ++it4) {
                        if (it4->first <= head || !is_PATH3_front[tid][it4->first] || record[tid][it4->first]) continue;
                        int cash4_x3 = (it4->second<<1)+it4->second;
                        int cash4_x5 = (it4->second<<2)+it4->second;
                        len_idx4 = idLen[it4->first];
                        if (check_cash(it3->second, cash3_x3, it4->second, cash4_x5)) {
                            for (int i = PATH3_INFO[tid][it4->first][0], end = PATH3_INFO[tid][it4->first][1]+PATH3_INFO[tid][it4->first][0]; i < end; ++i) {
                                if (!is_PATH3_back[tid][PATH3[tid][i].v] || record[tid][PATH3[tid][i].v] || record[tid][PATH3[tid][i].u]) continue;
                                int u = PATH3[tid][i].u;
                                int v = PATH3[tid][i].v;
                                int f = PATH3[tid][i].f;
                                int f_x5 = (f<<2)+f;
                                if (check_cash(it4->second, cash4_x3, f, f_x5)) {
#if ARM_NEON
                                    auto ptr = &res_char[tid][7][res_char_cnt_7];
                                    asm volatile("prfm pstl1keep, [%[p], #128] \n":[p]"+r"(ptr):);
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[head]);
                                    res_char_cnt_7 += len_head;
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it1->first]);
                                    res_char_cnt_7 += len_idx1;
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it2->first]);
                                    res_char_cnt_7 += len_idx2;
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it3->first]);
                                    res_char_cnt_7 += len_idx3;
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it4->first]);
                                    res_char_cnt_7 += len_idx4;
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[u]);
                                    res_char_cnt_7 += idLen[u];
                                    neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idLF[v]);
                                    res_char_cnt_7 += idLen[v];
                                    ++res_cnt_total;
#else
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[head], len_head);
                                    res_char_cnt_7 += len_head;
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it1->first], len_idx1);
                                    res_char_cnt_7 += len_idx1;
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it2->first], len_idx2);
                                    res_char_cnt_7 += len_idx2;
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it3->first], len_idx3);
                                    res_char_cnt_7 += len_idx3;
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it4->first], len_idx4);
                                    res_char_cnt_7 += len_idx4;
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idComma[u], idLen[u]);
                                    res_char_cnt_7 += idLen[u];
                                    memcpy(&res_char[tid][7][res_char_cnt_7], idLF[v], idLen[v]);
                                    res_char_cnt_7 += idLen[v];
                                    ++res_cnt_total;
#endif
                                }
                            }
                        }
                    }
                    record[tid][it3->first] = false;
                }
            }
            record[tid][it2->first] = false;
        }
    }
    record[tid][it1->first] = false;
    res_cnt[tid] = res_cnt_total;
    vtx_char_cnt[1][tid][head] += res_char_cnt_4 - res_char_cnt[tid][4];
    res_char_cnt[tid][4] = res_char_cnt_4;
    vtx_char_cnt[2][tid][head] += res_char_cnt_5 - res_char_cnt[tid][5];
    res_char_cnt[tid][5] = res_char_cnt_5;
    vtx_char_cnt[3][tid][head] += res_char_cnt_6 - res_char_cnt[tid][6];
    res_char_cnt[tid][6] = res_char_cnt_6;
    vtx_char_cnt[4][tid][head] += res_char_cnt_7 - res_char_cnt[tid][7];
    res_char_cnt[tid][7] = res_char_cnt_7;
}

void search_thread(int tid) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    int rec[PATH3_SIZE], v_infos[PATH3_SIZE*4], v_w_infos[PATH3_SIZE], w_list[PATH3_SIZE], v_list[PATH3_SIZE];
    int rec_sz = 0, v_infos_sz = 0, v_w_infos_sz = 0, w_list_sz = 0, v_list_sz = 0, PATH3_sz = 0, res3_cnt = 0;
    for (int idxi = 0; idxi < vtxCnt; ++idxi) {
        while(!atomic_compare_exchange_weak(&vtx_cnt_atomic, &idxi, vtx_cnt_atomic+1));
        if (idxi >= vtxCnt) break;
        if (outedgeSize[idxi] && outedges[idxi][outedgeSize[idxi]-1].first > idxi && inedgeSize[idxi] && inedges[idxi][inedgeSize[idxi]-1].first > idxi) {
            auto itj_end = inedges[idxi]+inedgeSize[idxi];
            for (auto itj = inedges[idxi]; itj != itj_end; ++itj) {
                if (itj->first <= idxi) continue;
                int last_sz = PATH3_sz;
                int cashj_x3 = (itj->second<<1)+itj->second;
                int cashj_x5 = (itj->second<<2)+itj->second;
                auto itk_end = inedges[itj->first]+inedgeSize[itj->first];
                for (auto itk = inedges[itj->first]; itk != itk_end; ++itk) {
                    if (itk->first <= idxi) continue;
                    int cashk_x3 = (itk->second<<1)+itk->second;
                    int cashk_x5 = (itk->second<<2)+itk->second;
                    if (!check_cash(itk->second, cashk_x3, itj->second, cashj_x5)) continue;
                    auto itm_end = inedges[itk->first]+inedgeSize[itk->first];
                    for (auto itm = inedges[itk->first]; itm != itm_end; ++itm) {
                        if (itm->first < idxi || itm->first == itj->first) continue;
                        int cashm_x3 = (itm->second<<1)+itm->second;
                        int cashm_x5 = (itm->second<<2)+itm->second;
                        if (!check_cash(itm->second, cashm_x3, itk->second, cashk_x5)) continue;
                        if (itm->first != idxi) {
                            PATH3[tid][PATH3_sz++] = Fragment(itm->first,itk->first,itj->first,itm->second);
                            if (++PATH3_INFO[tid][itm->first][1] == 1) rec[rec_sz++] = itm->first;
                        } else if (check_cash(itj->second, cashj_x3, itm->second, cashm_x5)) {
                            res3[tid][res3_cnt].path[0] = idxi;
                            res3[tid][res3_cnt].path[1] = itk->first;
                            res3[tid][res3_cnt++].path[2] = itj->first;
                            ++res_cnt[tid];
                        }
                    }
                }
                if (last_sz < PATH3_sz) {
                    v_infos[v_infos_sz++] = itj->first;
                    v_infos[v_infos_sz++] = itj->second;
                    v_infos[v_infos_sz++] = v_w_infos_sz;
                    v_infos[v_infos_sz++] = PATH3_sz-last_sz;
                    for (int i = last_sz; i < PATH3_sz; ++i) {
                        v_w_infos[v_w_infos_sz++] = PATH3[tid][i].w;
                    }
                }
            }
            if (rec_sz == 0) continue;
            sort(PATH3[tid], PATH3[tid]+PATH3_sz);
            for (int i = 0, i_end = PATH3_sz-1; i < i_end; ++i) {
                if (PATH3[tid][i].w != PATH3[tid][i+1].w) PATH3_INFO[tid][PATH3[tid][i+1].w][0] = i+1;
            }
            PATH3_INFO[tid][PATH3[tid][0].w][0] = 0;
            auto it1_end = outedges[idxi]+outedgeSize[idxi];
            for (auto it1 = outedges[idxi]; it1 != it1_end; ++it1) {
                if (it1->first <= idxi) continue;
                int cash1_x5 = (it1->second<<2)+it1->second;
                for (int i = 1; i < v_infos_sz; i+=4) {
                    int cashv_x3 = (v_infos[i]<<1)+v_infos[i];
                    if (check_cash(v_infos[i], cashv_x3, it1->second, cash1_x5)) {
                        v_list[v_list_sz++] = v_infos[i-1];
                        for (int j = v_infos[i+1], j_end = v_infos[i+1]+v_infos[i+2]; j < j_end; ++j) {
                            w_list[w_list_sz++] = v_w_infos[j];
                        }
                    }
                }
                if (w_list_sz == 0) continue;
                for (int i = 0; i < w_list_sz; ++i) is_PATH3_front[tid][w_list[i]] = true;
                for (int i = 0; i < v_list_sz; ++i) is_PATH3_back[tid][v_list[i]] = true;
                dfs(idxi, it1, tid);
                for (int i = 0; i < v_list_sz; ++i) is_PATH3_back[tid][v_list[i]] = false;
                for (int i = 0; i < w_list_sz; ++i) is_PATH3_front[tid][w_list[i]] = false;
                v_list_sz = 0;
                w_list_sz = 0;
            }
            for (int i = 0; i < rec_sz; ++i) {
                PATH3_INFO[tid][rec[i]][0] = 0;
                PATH3_INFO[tid][rec[i]][1] = 0;
            }
            rec_sz = 0;
            PATH3_sz = 0;
            v_infos_sz = 0;
            v_w_infos_sz = 0;
        }
    }
    sort(res3[tid], res3[tid]+res3_cnt);
    size_t last_cnt = 0, cur_cnt = 0;
    int head, path_k;
    for (int i = 0; i < res3_cnt; ++i) {
#if ARM_NEON
        head = res3[tid][i].path[0];
        neon_int8x8_memcpy(&res_char[tid][3][cur_cnt], idComma[head]);
        cur_cnt += idLen[head];
        path_k = res3[tid][i].path[1];
        neon_int8x8_memcpy(&res_char[tid][3][cur_cnt], idComma[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res3[tid][i].path[2];
        neon_int8x8_memcpy(&res_char[tid][3][cur_cnt], idLF[path_k]);
        cur_cnt += idLen[path_k];
        vtx_char_cnt[0][tid][head] += cur_cnt - last_cnt;
        last_cnt = cur_cnt;
#else
        head = res3[tid][i].path[0];
        memcpy(&res_char[tid][3][cur_cnt], idComma[head], idLen[head]);
        cur_cnt += idLen[head];
        path_k = res3[tid][i].path[1];
        memcpy(&res_char[tid][3][cur_cnt], idComma[path_k], idLen[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res3[tid][i].path[2];
        memcpy(&res_char[tid][3][cur_cnt], idLF[path_k], idLen[path_k]);
        cur_cnt += idLen[path_k];
        vtx_char_cnt[0][tid][head] += cur_cnt - last_cnt;
        last_cnt = cur_cnt;
#endif
    }
    res_char_cnt[tid][3] = cur_cnt;
    signed char* last_char_ptr;
    size_t last_char_cnt = 0;
    for (int i = MIN_DEPTH_LIMIT; i <= MAX_DEPTH_LIMIT; ++i) {
        last_char_ptr = res_char[tid][i];
        last_char_cnt = 0;
        for (int j = 0; j < vtxCnt; ++j) {
            if (vtx_char_cnt[i-MIN_DEPTH_LIMIT][tid][j] == 0) continue;
            vtx_char[i-MIN_DEPTH_LIMIT][j] = last_char_ptr + last_char_cnt;
            last_char_cnt += vtx_char_cnt[i-MIN_DEPTH_LIMIT][tid][j];
        }
    }
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the search time of thread " << tid << " : " << ds << " ms." << endl;
#endif
}

void search() {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) {
        res_char[tc][3] = res3_char[tc];
        res_char[tc][4] = res4_char[tc];
        res_char[tc][5] = res5_char[tc];
        res_char[tc][6] = res6_char[tc];
        res_char[tc][7] = res7_char[tc];
    }
    thread search_thd[SEARCH_THREAD_COUNT];
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc] = thread(search_thread, tc);
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc].join();

#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the search time : " << ds << " ms." << endl;
#endif
#if FREE
    free(shared_pool);
#endif
}

void store(const string &resultFile) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    size_t res_cnt_total = 0;
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) res_cnt_total += res_cnt[tc];
    size_t res_char_cnt_depth[8] = {0, 0, 0, 0, 0, 0, 0, 0}, res_char_total = 0;
    for (int i = MIN_DEPTH_LIMIT; i <= MAX_DEPTH_LIMIT; ++i) {
        for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) {
            res_char_cnt_depth[i] += res_char_cnt[tc][i];
            res_char_total += res_char_cnt[tc][i];
        }
    }
    string res_cnt_str = to_string(res_cnt_total) + '\n';
    size_t res_cnt_str_sz = res_cnt_str.size();
    int fd = open(resultFile.c_str(), O_RDWR | O_CREAT, 00777);
    lseek(fd, res_cnt_str_sz+res_char_total-1, SEEK_SET);
    write(fd, " ", 1);
    signed char *buf = (signed char*)mmap(NULL, res_cnt_str_sz+res_char_total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    signed char *buf_base = buf;
    close(fd);
    memcpy(buf_base, res_cnt_str.data(), res_cnt_str_sz);
    buf_base += res_cnt_str_sz;
    StoreInfo store_infos[STORE_PROC_COUNT];
    int offset_3_4 = res_char_cnt_depth[3] + res_char_cnt_depth[4];
    int offset_5_6 = res_char_cnt_depth[5] + res_char_cnt_depth[6];
    if (STORE_PROC_COUNT > 2) {
        store_infos[0].offset = 0;
        store_infos[0].len = 2;
        store_infos[0].end = vtxCnt;
        store_infos[0].beg = 0;
        store_infos[0].th[0] = 3;
        store_infos[0].th[1] = 4;

        store_infos[1].offset = offset_3_4;
        store_infos[1].len = 2;
        store_infos[1].end = vtxCnt;
        store_infos[1].beg = 0;
        store_infos[1].th[0] = 5;
        store_infos[1].th[1] = 6;

        size_t patchSize = res_char_cnt_depth[7] / (STORE_PROC_COUNT-2);

        store_infos[2].offset = store_infos[1].offset+offset_5_6;
        size_t res_char_count = store_infos[2].offset;
        store_infos[2].th[0] = 7;
        store_infos[2].len = 1;
        store_infos[2].beg = 0;
        int store_proc_count_sub1 = STORE_PROC_COUNT-1;
        for (int i = 3; i < STORE_PROC_COUNT; ++i) {
            store_infos[i].th[0] = 7;
            store_infos[i].len = 1;
            store_infos[i].offset = store_infos[i-1].offset+patchSize;
        }
        store_infos[store_proc_count_sub1].th[0] = 7;
        store_infos[store_proc_count_sub1].len = 1;
        store_infos[store_proc_count_sub1].end = vtxCnt;
        int j = 2;
        for (int i = 0; i < vtxCnt; ++i) {
            for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc){
                if (vtx_char_cnt[7-MIN_DEPTH_LIMIT][tc][i] == 0) continue;
                res_char_count += vtx_char_cnt[7-MIN_DEPTH_LIMIT][tc][i];
            }
            if (j < store_proc_count_sub1) {
                if (res_char_count >= store_infos[j+1].offset) {
                    store_infos[j].end = i + 1;
                    store_infos[j+1].offset = res_char_count;
                    store_infos[j+1].beg = i + 1;
                    ++j;
                }
            } else {
                break;
            }
        }
    } else if (STORE_PROC_COUNT == 2) {
        store_infos[0].offset = 0;
        store_infos[0].beg = 0;
        store_infos[0].end = vtxCnt;
        store_infos[0].len = 4;
        store_infos[0].th[0] = 3;
        store_infos[0].th[1] = 4;
        store_infos[0].th[2] = 5;
        store_infos[0].th[3] = 6;
        store_infos[1].len = 1;
        store_infos[0].beg = 0;
        store_infos[1].end = vtxCnt;
        store_infos[1].offset = offset_3_4 + offset_5_6;
        store_infos[1].th[0] = 7;
    } else {
        store_infos[0].offset = 0;
        int len = 0;
        for (int i = MIN_DEPTH_LIMIT; i <= MAX_DEPTH_LIMIT; ++i) {
            store_infos[0].th[len++] = i;
        }
        store_infos[0].len = len;
        store_infos[0].beg = 0;
        store_infos[0].end = vtxCnt;
    }
    int proc_id;
    pid_t pid = 1;
    for (int pc = 1; pc < STORE_PROC_COUNT; ++pc) {
        proc_id = pc;
        pid = fork();
        if (pid <= 0) break;
    }
    if (pid != 0) proc_id = 0;
    signed char *buffer = buf_base + store_infos[proc_id].offset;
    for (int i = 0; i < store_infos[proc_id].len; ++i) {
        auto &vtx_char_ = vtx_char[store_infos[proc_id].th[i]-MIN_DEPTH_LIMIT];
        auto &vtx_char_cnt_ = vtx_char_cnt[store_infos[proc_id].th[i]-MIN_DEPTH_LIMIT];
        for (int j = store_infos[proc_id].beg; j < store_infos[proc_id].end; ++j) {
            for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) {
                if (vtx_char_cnt_[tc][j] == 0) continue;
                memcpy(buffer, vtx_char_[j], vtx_char_cnt_[tc][j]);
                buffer += vtx_char_cnt_[tc][j];
            }
        }
    }
    if (proc_id > 0) exit(0);
#if STORE_WAIT
    int pc = 0;
    while (pc < STORE_PROC_COUNT-1) {
        pid = wait(NULL);
        if (pid > 0) ++pc;
    }
    munmap(buf, res_cnt_str_sz+res_char_total);
#endif
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the store time : " << ds << " ms." << endl;
    cout << "loop number : " << res_cnt_total << endl;
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
    int shmidInputs, inputSize, idMax;
    Edge *inputs;
    load(testFile, shmidInputs, inputs, inputSize, idMax);
    pretreat(shmidInputs, inputs, inputSize, idMax);
    search();
    store(resultFile);
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the elapsed time : " << ds << " ms." << endl;
#endif
    exit(0);
}
