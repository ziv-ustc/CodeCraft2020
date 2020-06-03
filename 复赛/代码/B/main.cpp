/*
    团队 : 守望&静望&观望
    赛区 ：上合赛区
    成绩 ：3.7189 (复赛B榜第二)
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
#define DATA_SET 47
#define MAX_EDGE_COUNT 2000000   // 6000000 2048*3000
#define MAX_RING_COUNT 4096*10 // 20000000
#define MAX_RING_CHAR_COUNT_3 3*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_4 4*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_5 5*11*MAX_RING_COUNT*4 // 0.4
#define MAX_RING_CHAR_COUNT_6 6*11*MAX_RING_COUNT*7 // 0.7
#define MAX_RING_CHAR_COUNT_7 7*11*MAX_RING_COUNT*9 // 0.9
#define MAX_RING_CHAR_COUNT_8 8*11*MAX_RING_COUNT*10 // 1.0 
#define PATH4_SIZE 100000 
#define MAX_VTX_COUNT 4000000
#define MIN_DEPTH_LIMIT 3
#define MAX_DEPTH_LIMIT 8 
#define LOAD_PROC_COUNT 4
#define PRETREAT_THREAD_COUNT 4
#define SEARCH_THREAD_COUNT 4
#define STORE_PROC_COUNT 5

#define Y_MIN 429496729 // INT_MAX/5
#define X_MIN 715827882 // INT_MAX/3

struct Fragment { // !!!!!!!!!
    int w;
    int x;
    int u;
    int v;
    size_t f;
    Fragment(): w(0), x(0), u(0), v(0), f(0) { }
    Fragment(int &w, int &x, int &u, int &v, size_t &f): w(w), x(x), u(u), v(v), f(f) { }
    bool operator<(Fragment &rhs) {
        if (w != rhs.w) return w < rhs.w;
        if (x != rhs.x) return x < rhs.x;
        if (u != rhs.u) return u < rhs.u;
        return v < rhs.v;
    }
};

struct Transfer {
    int idx;
    size_t cash;
    Transfer(): idx(0), cash(0) { }
    Transfer(int i, size_t c): idx(i), cash(c) { }
    bool operator<(const Transfer &rhs) const {
        return idx < rhs.idx;
    }
};

struct Ring {
    int path[4];
    bool operator<(const Ring &rhs) {
        if (path[0] != rhs.path[0]) return path[0] < rhs.path[0];
        if (path[1] != rhs.path[1]) return path[1] < rhs.path[1];
        if (path[2] != rhs.path[2]) return path[2] < rhs.path[2];
        if (path[3] != rhs.path[3]) return path[3] < rhs.path[3]; 
        return true;
    }
};

struct Edge {
    int id1;
    int id2;
    size_t cash;
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

#define check_cash(X,X_3,Y,Y_5) (( X <= Y_5)&&( Y <= X_3))

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
    int id_max = 0;
    size_t val = 0;
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
        ++i;
        while (buf[i] >= '0') val = (val<<3) + (val<<1) + ASCII[buf[i++]];
        int add_zero = 2;
        switch (buf[i]) {
            case '.': {
                ++i;
                while (buf[i] >= '0') {
                    val = (val<<3) + (val<<1) + ASCII[buf[i++]];
                    --add_zero;
                };
                break;
            }
            default: {
                add_zero = 2;
                break;
            }
        }
        while (add_zero--) val = (val<<3) + (val<<1);
        inputs[sz1].cash = val;
        inputs[sz2].cash = val;
        sz1 += 2;
        val = 0;
        while (buf[i++] != '\n');
    }
    inputs[max_data_size_xload+proc_id].id1 = sz1-offset;
    sort(inputs+offset, inputs+sz1);
    inputs[max_data_size_xload_1+proc_id].id1 = id_max;
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
pair<int,size_t> shared_pool[MAX_VTX_COUNT];
int outedges[MAX_VTX_COUNT], inedges[MAX_VTX_COUNT];
int outedgeSize[MAX_VTX_COUNT], inedgeSize[MAX_VTX_COUNT];
signed char idComma[MAX_VTX_COUNT][16];
signed char idLF[MAX_VTX_COUNT][16];
int idLen[MAX_VTX_COUNT];

void pretreat(int &shmidInputs, Edge* &inputs, int &inputSize, int &idMax) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    int sizeof_int = sizeof(int), sizeof_edge = sizeof(Edge), sizeof_intx64 = sizeof_int*64;
    int max_data_size_xload = LOAD_PROC_COUNT * MAX_EDGE_COUNT * 2;
    Edge *input_ptr[LOAD_PROC_COUNT];
    int input_sz[LOAD_PROC_COUNT], offset[LOAD_PROC_COUNT];
    for (int pc = 0, last = 0; pc < LOAD_PROC_COUNT; ++pc) {
        input_ptr[pc] = inputs + pc * MAX_EDGE_COUNT * 2;
        input_sz[pc] = inputs[max_data_size_xload+pc].id1;
        offset[pc] = last;
        last = offset[pc] + input_sz[pc];
    }
    Edge *input = (Edge*)malloc(inputSize*sizeof_edge);
    Edge *input_tmp = (Edge*)malloc(inputSize*sizeof_edge);
    auto merge_func = [&](int id) {
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
    int idtoidx_sz = (inputSize>>1) < (idMax) ? (inputSize>>1) : (idMax);
    int *idtoidx = (int*)malloc(sizeof_intx64*idtoidx_sz);
    int bitmap_sz = (idMax>>6) + 1;
    int *idtoidx_beg = (int*)malloc(bitmap_sz*sizeof_int);
    memset(idtoidx_beg, -1, bitmap_sz*sizeof_int);
    int *input_beg = (int*)malloc((inputSize+1)*sizeof_int);
    idtoidx_sz = 0;
    for (int i = 0, j = 0; i < inputSize; i = j) {
        id = input[i].id1;
        id_high = id >> 6;
        id_low = id & 63;
        input_beg[vtx_cnt] = i;
        if (idtoidx_beg[id_high] == -1) {
            idtoidx_beg[id_high] = idtoidx_sz;
            idtoidx_sz += 64;
        }
        idtoidx[idtoidx_beg[id_high]+id_low] = vtx_cnt;
        while (j < inputSize && input[j].id1 == id) {
            if (input[j].isInedge) ++inedgeSize[vtx_cnt];
            else ++outedgeSize[vtx_cnt];
            ++j;
        }
        ++vtx_cnt;
    }
    input_beg[vtx_cnt] = inputSize;
    uint64_t input_offset[8] = {0};
    {
        uint64_t num1[2], num2[2], num3[2], num4[2];
        const int vtx_cnt1 = vtx_cnt/4, vtx_cnt2 = vtx_cnt/2, vtx_cnt3 = 3*vtx_cnt/4;
        auto add_func = [](const int &beg, const int &end, uint64_t *num) {
            num[0] = 0;
            num[1] = 0;
            for (int i = beg; i < end; ++i) {
                num[0] += inedgeSize[i];
                num[1] += outedgeSize[i];
            }
        };
        thread th0(add_func,0,vtx_cnt1,num1);
        thread th1(add_func,vtx_cnt1,vtx_cnt2,num2);
        thread th2(add_func,vtx_cnt2,vtx_cnt3,num3);
        thread th3(add_func,vtx_cnt3,vtx_cnt,num4);
        th0.join();
        th1.join();
        th2.join();
        th3.join();
        input_offset[1] = num1[0];
        input_offset[2] = input_offset[1] + num2[0];
        input_offset[3] = input_offset[2] + num3[0];
        input_offset[4] = input_offset[3] + num4[0];
        input_offset[5] = input_offset[4] + num1[1];
        input_offset[6] = input_offset[5] + num2[1];
        input_offset[7] = input_offset[6] + num3[1];
    }
    int *idxtoid = (int*)malloc(vtx_cnt*sizeof_int);
    auto create_func = [&](const int &beg, const int &end, int id) {
        int sz1, sz2;
        int input_beg_;
        int in_offset = input_offset[id];
        int out_offset = input_offset[id+4];
        for (int i = beg, j = beg; i < end; ++i) {
            input_beg_ = input_beg[i];
            id = input[input_beg_].id1;
            idxtoid[i] = id;
            outedges[i] = out_offset;
            out_offset += outedgeSize[i];
            inedges[i] = in_offset;
            in_offset += inedgeSize[i];
            sz1 = 0;
            sz2 = 0;
            j = input_beg_;
            while (j < input_beg[i+1]) {
                if (input[j].isInedge) {
                    int id = inedges[i] + sz1;
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    shared_pool[id].first = int(idtoidx[idtoidx_beg[high]+low]);
                    shared_pool[id].second = size_t(input[j++].cash);
                    ++sz1;
                } else {
                    int id = outedges[i] + sz2;
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    shared_pool[id].first = int(idtoidx[idtoidx_beg[high]+low]);
                    shared_pool[id].second = size_t(input[j++].cash);
                    ++sz2;
                }
            }
        }
    };
    {
        const int vtx_cnt1 = vtx_cnt/4, vtx_cnt2 = vtx_cnt/2, vtx_cnt3 = 3*vtx_cnt/4;
        thread th0(create_func,0,vtx_cnt1,0);
        thread th1(create_func,vtx_cnt1,vtx_cnt2,1);
        thread th2(create_func,vtx_cnt2,vtx_cnt3,2);
        thread th3(create_func,vtx_cnt3,vtx_cnt,3);
        th0.join();
        th1.join();
        th2.join();
        th3.join();
    }
    auto transfer_func = [&](PretreatInfo *infos, int tc) {
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
            auto shared_out = shared_pool + outedges[i];
            auto shared_in = shared_pool + inedges[i];
            sort(shared_out, shared_out+outedgeSize[i], [](pair<int,size_t>&a, pair<int,size_t>&b)->bool{return a.first < b.first;});
            sort(shared_in, shared_in+inedgeSize[i], [](pair<int,size_t>&a, pair<int,size_t>&b)->bool{return a.first < b.first;});
        }
#if TEST
        auto et = std::chrono::steady_clock::now();
        double ds = std::chrono::duration<double>(et-st).count()*1000.0;
        cout << "the pretreat time of thread " << tc << " : " << ds << " ms." << endl;
#endif
    };
    PretreatInfo pret_infos[PRETREAT_THREAD_COUNT];
    for (int tc = 0, last = 0, cur = 0; tc < PRETREAT_THREAD_COUNT; ++tc) {
        pret_infos[tc].l = last;
        cur = (tc+1) * vtx_cnt / PRETREAT_THREAD_COUNT;
        pret_infos[tc].r = cur;
        last = cur;
    }
    thread pret_thd[PRETREAT_THREAD_COUNT];
    for (int tc = 0; tc < PRETREAT_THREAD_COUNT; ++tc) pret_thd[tc] = thread(transfer_func, &pret_infos[tc], tc);
    for (int tc = 0; tc < PRETREAT_THREAD_COUNT; ++tc) pret_thd[tc].join();
    vtxCnt = vtx_cnt;
#if FREE_SHM
    shmdt(inputs);
    shmctl(shmidInputs, IPC_RMID, NULL);
#endif
#if FREE
    free(input_beg);
    free(idxtoid);
    free(idtoidx_beg);
    free(input);
    free(input_tmp);
    free(idtoidx);
#endif
#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the pretreat time : " << ds << " ms." << endl;
#endif
}

signed char res_char_[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_3+MAX_RING_CHAR_COUNT_4+MAX_RING_CHAR_COUNT_5
                                           +MAX_RING_CHAR_COUNT_6+MAX_RING_CHAR_COUNT_7+MAX_RING_CHAR_COUNT_8];
signed char *res_char[SEARCH_THREAD_COUNT][9];            
size_t res_char_cnt[SEARCH_THREAD_COUNT][9] = {0};         
signed char *vtx_char[6][MAX_VTX_COUNT];                  
size_t vtx_char_cnt[6][SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
Ring res3[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_3/33];
Ring res4[SEARCH_THREAD_COUNT][MAX_RING_CHAR_COUNT_4/44];
int res_cnt[SEARCH_THREAD_COUNT] = {0};
Fragment PATH4[SEARCH_THREAD_COUNT][PATH4_SIZE];       
int PATH4_INFO[SEARCH_THREAD_COUNT][MAX_VTX_COUNT][2];
bool record[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
bool is_PATH4_front[SEARCH_THREAD_COUNT][MAX_VTX_COUNT]; 
bool is_PATH4_back[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
atomic<int> vtx_cnt_atomic(0);

void dfs(int head, pair<int,size_t>* it1, int tid) {
    size_t res_char_cnt_5 = res_char_cnt[tid][5], res_char_cnt_6 = res_char_cnt[tid][6];
    size_t res_char_cnt_7 = res_char_cnt[tid][7], res_char_cnt_8 = res_char_cnt[tid][8];
    int len_head = idLen[head], len_idx1, len_idx2, len_idx3, len_idx4;
    int res_cnt_total = res_cnt[tid];
    size_t cash1_x3 = (it1->second<<1)+it1->second;
    len_idx1 = idLen[it1->first];
    if (is_PATH4_front[tid][it1->first]) {
        for (int i = PATH4_INFO[tid][it1->first][0], end = PATH4_INFO[tid][it1->first][1]+PATH4_INFO[tid][it1->first][0]; i < end; ++i) {
            if (!is_PATH4_back[tid][PATH4[tid][i].v]) continue;
            int x = PATH4[tid][i].x;
            int u = PATH4[tid][i].u;
            int v = PATH4[tid][i].v; 
            size_t f = PATH4[tid][i].f;
            size_t f_x5 = (f<<2)+f;
            if (check_cash(it1->second, cash1_x3, f, f_x5)) {
#if ARM_NEON
                neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[head]);
                res_char_cnt_5 += len_head;
                neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[it1->first]);
                res_char_cnt_5 += len_idx1;
                neon_int8x8_memcpy(&res_char[tid][5][res_char_cnt_5], idComma[x]);
                res_char_cnt_5 += idLen[x];
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
                memcpy(&res_char[tid][5][res_char_cnt_5], idComma[x], idLen[x]);
                res_char_cnt_5 += idLen[x];
                memcpy(&res_char[tid][5][res_char_cnt_5], idComma[u], idLen[u]);
                res_char_cnt_5 += idLen[u];
                memcpy(&res_char[tid][5][res_char_cnt_5], idLF[v], idLen[v]);
                res_char_cnt_5 += idLen[v];
                ++res_cnt_total;
#endif
            }
        }
    }
    record[tid][it1->first] = true;
    auto it2 = shared_pool+outedges[it1->first];
    auto it2_end =it2+outedgeSize[it1->first];
    for (; it2 != it2_end; ++it2) {
        if (it2->first <= head) continue;
        size_t cash2_x3 = (it2->second<<1)+it2->second;
        size_t cash2_x5 = (it2->second<<2)+it2->second;
        len_idx2 = idLen[it2->first];
        if (check_cash(it1->second, cash1_x3, it2->second, cash2_x5)) {
            if (is_PATH4_front[tid][it2->first]) { 
                for (int i = PATH4_INFO[tid][it2->first][0], end = PATH4_INFO[tid][it2->first][1]+PATH4_INFO[tid][it2->first][0]; i < end; ++i) {
                    if ( !is_PATH4_back[tid][PATH4[tid][i].v] || record[tid][PATH4[tid][i].x]
                         || record[tid][PATH4[tid][i].u] || record[tid][PATH4[tid][i].v] ) continue; 
                    int x = PATH4[tid][i].x;
                    int u = PATH4[tid][i].u; 
                    int v = PATH4[tid][i].v; 
                    size_t f = PATH4[tid][i].f;
                    size_t f_x5 = (f<<2)+f;
                    if (check_cash(it2->second, cash2_x3, f, f_x5)) {
#if ARM_NEON
                        neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[head]);
                        res_char_cnt_6 += len_head;
                        neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it1->first]);
                        res_char_cnt_6 += len_idx1;
                        neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[it2->first]);
                        res_char_cnt_6 += len_idx2;
                        neon_int8x8_memcpy(&res_char[tid][6][res_char_cnt_6], idComma[x]);
                        res_char_cnt_6 += idLen[x];
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
                        memcpy(&res_char[tid][6][res_char_cnt_6], idComma[x], idLen[x]);
                        res_char_cnt_6 += idLen[x];
                        memcpy(&res_char[tid][6][res_char_cnt_6], idComma[u], idLen[u]);
                        res_char_cnt_6 += idLen[u];
                        memcpy(&res_char[tid][6][res_char_cnt_6], idLF[v], idLen[v]);
                        res_char_cnt_6 += idLen[v];
                        ++res_cnt_total;
#endif
                    }
                }
            }
            record[tid][it2->first] = true;
            auto it3 =shared_pool+ outedges[it2->first];
            auto it3_end = it3+outedgeSize[it2->first];
            for (; it3 != it3_end; ++it3) {
                if (it3->first <= head || it3->first == it1->first) continue;
                size_t cash3_x3 = (it3->second<<1)+it3->second;
                size_t cash3_x5 = (it3->second<<2)+it3->second;
                len_idx3 = idLen[it3->first];
                if (check_cash(it2->second, cash2_x3, it3->second, cash3_x5)) {
                    if (is_PATH4_front[tid][it3->first]) {
                        for (int i = PATH4_INFO[tid][it3->first][0], end = PATH4_INFO[tid][it3->first][1]+PATH4_INFO[tid][it3->first][0]; i < end; ++i) {
                            if ( !is_PATH4_back[tid][PATH4[tid][i].v] || record[tid][PATH4[tid][i].x]
                                 || record[tid][PATH4[tid][i].v] || record[tid][PATH4[tid][i].u]) continue; 
                            int x = PATH4[tid][i].x;
                            int u = PATH4[tid][i].u;
                            int v = PATH4[tid][i].v; 
                            size_t f = PATH4[tid][i].f; 
                            size_t f_x5 = (f<<2)+f;
                            if (check_cash(it3->second, cash3_x3, f, f_x5)) { 
#if ARM_NEON
                                auto ptr = &res_char[tid][7][res_char_cnt_8];
                                asm volatile("prfm pstl1keep, [%[p], #128] \n":[p]"+r"(ptr):);
                                neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[head]);
                                res_char_cnt_7 += len_head;
                                neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it1->first]);
                                res_char_cnt_7 += len_idx1;
                                neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it2->first]);
                                res_char_cnt_7 += len_idx2;
                                neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[it3->first]);
                                res_char_cnt_7 += len_idx3;
                                neon_int8x8_memcpy(&res_char[tid][7][res_char_cnt_7], idComma[x]);
                                res_char_cnt_7 += idLen[x];
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
                                memcpy(&res_char[tid][7][res_char_cnt_7], idComma[x], idLen[x]);
                                res_char_cnt_7 += idLen[x];
                                memcpy(&res_char[tid][7][res_char_cnt_7], idComma[u], idLen[u]);
                                res_char_cnt_7 += idLen[u];
                                memcpy(&res_char[tid][7][res_char_cnt_7], idLF[v], idLen[v]);
                                res_char_cnt_7 += idLen[v];
                                ++res_cnt_total;
#endif
                            }
                        }
                    }
                    record[tid][it3->first] = true;
                    auto it4 =shared_pool+ outedges[it3->first];
                    auto it4_end = it4+outedgeSize[it3->first];
                    for (; it4 != it4_end; ++it4) {
                        if (it4->first <= head || !is_PATH4_front[tid][it4->first] || record[tid][it4->first]) continue;
                        size_t cash4_x3 = (it4->second<<1)+it4->second;
                        size_t cash4_x5 = (it4->second<<2)+it4->second;
                        len_idx4 = idLen[it4->first];
                        if (check_cash(it3->second, cash3_x3, it4->second, cash4_x5)) {
                            for (int i = PATH4_INFO[tid][it4->first][0], end = PATH4_INFO[tid][it4->first][1]+PATH4_INFO[tid][it4->first][0]; i < end; ++i) {
                                if ( !is_PATH4_back[tid][PATH4[tid][i].v] || record[tid][PATH4[tid][i].x]
                                     || record[tid][PATH4[tid][i].v] || record[tid][PATH4[tid][i].u] ) continue;
                                int x = PATH4[tid][i].x;
                                int u = PATH4[tid][i].u;
                                int v = PATH4[tid][i].v;
                                size_t f = PATH4[tid][i].f;
                                size_t f_x5 = (f<<2)+f;
                                if (check_cash(it4->second, cash4_x3, f, f_x5)) {
#if ARM_NEON
                                    auto ptr = &res_char[tid][8][res_char_cnt_8];
                                    asm volatile("prfm pstl1keep, [%[p], #512] \n":[p]"+r"(ptr):);
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[head]);
                                    res_char_cnt_8 += len_head;
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it1->first]);
                                    res_char_cnt_8 += len_idx1;
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it2->first]);
                                    res_char_cnt_8 += len_idx2;
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it3->first]);
                                    res_char_cnt_8 += len_idx3;
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it4->first]);
                                    res_char_cnt_8 += len_idx4;
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[x]);
                                    res_char_cnt_8 += idLen[x];
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idComma[u]);
                                    res_char_cnt_8 += idLen[u];
                                    neon_int8x8_memcpy(&res_char[tid][8][res_char_cnt_8], idLF[v]);
                                    res_char_cnt_8 += idLen[v];
                                    ++res_cnt_total;
#else
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[head], len_head);
                                    res_char_cnt_8 += len_head;
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it1->first], len_idx1);
                                    res_char_cnt_8 += len_idx1;
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it2->first], len_idx2);
                                    res_char_cnt_8 += len_idx2;
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it3->first], len_idx3);
                                    res_char_cnt_8 += len_idx3;
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[it4->first], len_idx4);
                                    res_char_cnt_8 += len_idx4;
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[x], idLen[x]);
                                    res_char_cnt_8 += idLen[x];
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idComma[u], idLen[u]);
                                    res_char_cnt_8 += idLen[u];
                                    memcpy(&res_char[tid][8][res_char_cnt_8], idLF[v], idLen[v]);
                                    res_char_cnt_8 += idLen[v];
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
    vtx_char_cnt[2][tid][head] += res_char_cnt_5 - res_char_cnt[tid][5];
    res_char_cnt[tid][5] = res_char_cnt_5;                               
    vtx_char_cnt[3][tid][head] += res_char_cnt_6 - res_char_cnt[tid][6];
    res_char_cnt[tid][6] = res_char_cnt_6;                               
    vtx_char_cnt[4][tid][head] += res_char_cnt_7 - res_char_cnt[tid][7];
    res_char_cnt[tid][7] = res_char_cnt_7;                              
    vtx_char_cnt[5][tid][head] += res_char_cnt_8 - res_char_cnt[tid][8];
    res_char_cnt[tid][8] = res_char_cnt_8;
}

void search_thread(int tid) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    size_t rec[PATH4_SIZE], v_infos[PATH4_SIZE*4], v_w_infos[PATH4_SIZE], w_list[PATH4_SIZE], v_list[PATH4_SIZE];
    int rec_sz = 0, v_infos_sz = 0, v_w_infos_sz = 0, w_list_sz = 0, v_list_sz = 0, PATH4_sz = 0, res3_cnt = 0, res4_cnt = 0;
    for (int idxi = 0; idxi < vtxCnt; ++idxi) {
        while(!atomic_compare_exchange_weak(&vtx_cnt_atomic, &idxi, vtx_cnt_atomic+1));
        if (idxi >= vtxCnt) break;
        if (outedgeSize[idxi] &&shared_pool[outedges[idxi]+outedgeSize[idxi]-1].first > idxi && inedgeSize[idxi] && shared_pool[inedges[idxi]+inedgeSize[idxi]-1].first > idxi) {
            auto itj = shared_pool+inedges[idxi];
            auto itj_end =itj+inedgeSize[idxi];
            for (; itj != itj_end; ++itj) {
                if (itj->first <= idxi) continue;
                int last_sz = PATH4_sz;
                size_t cashj_x3 = (itj->second<<1)+itj->second;
                size_t cashj_x5 = (itj->second<<2)+itj->second;
                auto itk = shared_pool+inedges[itj->first];
                auto itk_end = itk+inedgeSize[itj->first];
                for (; itk != itk_end; ++itk)  {
                    if (itk->first <= idxi) continue;
                    size_t cashk_x3 = (itk->second<<1)+itk->second;
                    size_t cashk_x5 = (itk->second<<2)+itk->second;
                    if (!check_cash(itk->second, cashk_x3, itj->second, cashj_x5)) continue;
                    auto itm =shared_pool+inedges[itk->first];
                    auto itm_end = itm+inedgeSize[itk->first];
                    for (; itm != itm_end; ++itm)  {
                        if (itm->first < idxi || itm->first == itj->first) continue;
                        size_t cashm_x3 = (itm->second<<1)+itm->second;
                        size_t cashm_x5 = (itm->second<<2)+itm->second;
                        if (!check_cash(itm->second, cashm_x3, itk->second, cashk_x5)) continue;
                        if (itm->first == idxi) { 
                            if (check_cash(itj->second, cashj_x3, itm->second, cashm_x5)) {
                                res3[tid][res3_cnt].path[0] = idxi;
                                res3[tid][res3_cnt].path[1] = itk->first;
                                res3[tid][res3_cnt++].path[2] = itj->first;
                                ++res_cnt[tid];
                            }
                            continue;
                        }
                        auto itn =shared_pool+ inedges[itm->first];
                        auto itn_end = itn+inedgeSize[itm->first];    
                        for (; itn != itn_end; ++itn) {  
                            if (itn->first < idxi || itn->first == itj->first || itn->first == itk->first) continue;
                            size_t cashn_x3 = (itn->second<<1)+itn->second;
                            size_t cashn_x5 = (itn->second<<2)+itn->second;
                            if (!check_cash(itn->second, cashn_x3, itm->second, cashm_x5)) continue;
                            if (itn->first != idxi) {
                                PATH4[tid][PATH4_sz++] = Fragment(itn->first,itm->first,itk->first,itj->first,itn->second);
                                if (++PATH4_INFO[tid][itn->first][1] == 1) rec[rec_sz++] = itn->first;
                            } else if (check_cash(itj->second, cashj_x3, itn->second, cashn_x5)) {
                                res4[tid][res4_cnt].path[0] = idxi;
                                res4[tid][res4_cnt].path[1] = itm->first;
                                res4[tid][res4_cnt].path[2] = itk->first;
                                res4[tid][res4_cnt++].path[3] = itj->first;
                                ++res_cnt[tid];
                            }
                        }
                    }
                }
                if (last_sz < PATH4_sz) {
                    v_infos[v_infos_sz++] = itj->first;
                    v_infos[v_infos_sz++] = itj->second;
                    v_infos[v_infos_sz++] = v_w_infos_sz;
                    v_infos[v_infos_sz++] = PATH4_sz-last_sz;
                    for (int i = last_sz; i < PATH4_sz; ++i) { 
                        v_w_infos[v_w_infos_sz++] = PATH4[tid][i].w; 
                    }
                }
            }
            if (rec_sz == 0) continue;
            sort(PATH4[tid], PATH4[tid]+PATH4_sz);
            for (int i = 0, i_end = PATH4_sz-1; i < i_end; ++i) {
                if (PATH4[tid][i].w != PATH4[tid][i+1].w) PATH4_INFO[tid][PATH4[tid][i+1].w][0] = i+1;
            }
            PATH4_INFO[tid][PATH4[tid][0].w][0] = 0; 
            auto it1 = shared_pool+outedges[idxi];
            auto it1_end = it1+outedgeSize[idxi];
            for (; it1 != it1_end; ++it1) {
                if (it1->first <= idxi) continue;
                size_t cash1_x5 = (it1->second<<2)+it1->second;
                for (int i = 1; i < v_infos_sz; i+=4) {
                    size_t cashv_x3 = (v_infos[i]<<1)+v_infos[i];
                    if (check_cash(v_infos[i], cashv_x3, it1->second, cash1_x5)) {
                        v_list[v_list_sz++] = v_infos[i-1];
                        for (int j = v_infos[i+1], j_end = v_infos[i+1]+v_infos[i+2]; j < j_end; ++j) {
                            w_list[w_list_sz++] = v_w_infos[j];
                        }
                    }
                }
                if (w_list_sz == 0) continue;
                for (int i = 0; i < w_list_sz; ++i) is_PATH4_front[tid][w_list[i]] = true; 
                for (int i = 0; i < v_list_sz; ++i) is_PATH4_back[tid][v_list[i]] = true; 
                dfs(idxi, it1, tid);
                for (int i = 0; i < v_list_sz; ++i) is_PATH4_back[tid][v_list[i]] = false; 
                for (int i = 0; i < w_list_sz; ++i) is_PATH4_front[tid][w_list[i]] = false; 
                v_list_sz = 0;
                w_list_sz = 0;
            }
            for (int i = 0; i < rec_sz; ++i) {
                PATH4_INFO[tid][rec[i]][0] = 0; 
                PATH4_INFO[tid][rec[i]][1] = 0;
            }
            rec_sz = 0;
            PATH4_sz = 0;
            v_infos_sz = 0;
            v_w_infos_sz = 0;
        }
    }
    sort(res3[tid], res3[tid]+res3_cnt);
    sort(res4[tid], res4[tid]+res4_cnt); 
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
    cur_cnt = 0; 
    last_cnt = 0; 
    for (int i = 0; i < res4_cnt; ++i) { 
#if ARM_NEON
        head = res4[tid][i].path[0];
        neon_int8x8_memcpy(&res_char[tid][4][cur_cnt], idComma[head]);
        cur_cnt += idLen[head];
        path_k = res4[tid][i].path[1];
        neon_int8x8_memcpy(&res_char[tid][4][cur_cnt], idComma[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res4[tid][i].path[2];
        neon_int8x8_memcpy(&res_char[tid][4][cur_cnt], idComma[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res4[tid][i].path[3];
        neon_int8x8_memcpy(&res_char[tid][4][cur_cnt], idLF[path_k]);
        cur_cnt += idLen[path_k];
        vtx_char_cnt[1][tid][head] += cur_cnt - last_cnt;
        last_cnt = cur_cnt;
#else
        head = res4[tid][i].path[0];
        memcpy(&res_char[tid][4][cur_cnt], idComma[head], idLen[head]);
        cur_cnt += idLen[head];
        path_k = res4[tid][i].path[1];
        memcpy(&res_char[tid][4][cur_cnt], idComma[path_k], idLen[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res4[tid][i].path[2];
        memcpy(&res_char[tid][4][cur_cnt], idComma[path_k], idLen[path_k]);
        cur_cnt += idLen[path_k];
        path_k = res4[tid][i].path[3];
        memcpy(&res_char[tid][4][cur_cnt], idLF[path_k], idLen[path_k]);
        cur_cnt += idLen[path_k];
        vtx_char_cnt[1][tid][head] += cur_cnt - last_cnt;
        last_cnt = cur_cnt;
#endif
    }
    res_char_cnt[tid][4] = cur_cnt;
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
        res_char[tc][3] = &res_char_[tc][0];
        res_char[tc][4] = &res_char_[tc][MAX_RING_CHAR_COUNT_3];
        res_char[tc][5] = &res_char_[tc][MAX_RING_CHAR_COUNT_3+MAX_RING_CHAR_COUNT_4];
        res_char[tc][6] = &res_char_[tc][MAX_RING_CHAR_COUNT_3+MAX_RING_CHAR_COUNT_4+MAX_RING_CHAR_COUNT_5];
        res_char[tc][7] = &res_char_[tc][MAX_RING_CHAR_COUNT_3+MAX_RING_CHAR_COUNT_4+MAX_RING_CHAR_COUNT_5+MAX_RING_CHAR_COUNT_6];
        res_char[tc][8] = &res_char_[tc][MAX_RING_CHAR_COUNT_3+MAX_RING_CHAR_COUNT_4+MAX_RING_CHAR_COUNT_5+MAX_RING_CHAR_COUNT_6+MAX_RING_CHAR_COUNT_7]; 
    }
    thread search_thd[SEARCH_THREAD_COUNT];
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc] = thread(search_thread, tc);
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc].join();

#if TEST
    auto et = std::chrono::steady_clock::now();
    double ds = std::chrono::duration<double>(et-st).count()*1000.0;
    cout << "the search time : " << ds << " ms." << endl;
#endif
}

void store(const string &resultFile) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    size_t res_cnt_total = 0;
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) res_cnt_total += res_cnt[tc];
    size_t res_char_cnt_depth[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, res_char_total = 0;
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
    int offset_3 = res_char_cnt_depth[3];
    int offset_3_4 = res_char_cnt_depth[3] + res_char_cnt_depth[4];
    int offset_4_5_6_7 = res_char_cnt_depth[4]+res_char_cnt_depth[5] + res_char_cnt_depth[6]+res_char_cnt_depth[7];
    if (STORE_PROC_COUNT > 2) {
        store_infos[0].offset = 0;
        store_infos[0].len = 1;
        store_infos[0].end = vtxCnt;
        store_infos[0].beg = 0;
        store_infos[0].th[0] = 3;

        store_infos[1].offset = offset_3;
        store_infos[1].len = 4;
        store_infos[1].end = vtxCnt;
        store_infos[1].beg = 0;
        store_infos[1].th[0] = 4;
        store_infos[1].th[1] = 5;
        store_infos[1].th[2] = 6;
        store_infos[1].th[3] = 7;
        size_t patchSize = res_char_cnt_depth[8] / (STORE_PROC_COUNT-2);

        store_infos[2].offset = store_infos[1].offset+offset_4_5_6_7;
        size_t res_char_count = store_infos[2].offset;
        store_infos[2].th[0] = 8;
        store_infos[2].len = 1;
        store_infos[2].beg = 0;
        int store_proc_count_sub1 = STORE_PROC_COUNT-1;
        for (int i = 3; i < STORE_PROC_COUNT; ++i) {
            store_infos[i].th[0] = 8;
            store_infos[i].len = 1;
            store_infos[i].offset = store_infos[i-1].offset+patchSize;
        }
        store_infos[store_proc_count_sub1].th[0] = 8;
        store_infos[store_proc_count_sub1].len = 1;
        store_infos[store_proc_count_sub1].end = vtxCnt;
        int j = 2;
        for (int i = 0; i < vtxCnt; ++i) {
            for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc){
                if (vtx_char_cnt[8-MIN_DEPTH_LIMIT][tc][i] == 0) continue;
                res_char_count += vtx_char_cnt[8-MIN_DEPTH_LIMIT][tc][i];
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
        store_infos[0].len = 5;
        store_infos[0].th[0] = 3;
        store_infos[0].th[1] = 4;
        store_infos[0].th[2] = 5;
        store_infos[0].th[3] = 6;
        store_infos[0].th[4] = 7;
        store_infos[1].len = 1;
        store_infos[0].beg = 0;
        store_infos[1].end = vtxCnt;
        store_infos[1].offset = offset_3_4 + offset_4_5_6_7;
        store_infos[1].th[0] = 8;
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
#if TEST
    {auto et = std::chrono::steady_clock::now();
        double ds = std::chrono::duration<double>(et-st).count()*1000.0;
        cout << "the store time of process : "<<proc_id<<" " << ds << " ms." << endl;}
#endif
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
