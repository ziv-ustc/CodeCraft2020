/*
    团队 : 守望&静望&观望
    赛区 ：上合赛区
    成绩 ：346.6120 (决赛A榜第十一)
    备注 ：本代码在346s的基础上再次优化，保守估计能达到320s以内
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
// open : 1 2 3 4 5 6 7 8 9 10 11 12 std : 13 14 15 16
#define DATA_SET 15
#define MAX_EDGE_COUNT 2500000
#define MAX_VTX_COUNT 2500000
#define OUTPUT_LIMIT 100
#define LOAD_PROC_COUNT 4
#define PRETREAT_THREAD_COUNT 4
#define SEARCH_THREAD_COUNT 8
#define PREV_CNT 1

typedef uint8_t CashType;
typedef uint16_t DistType;
#define DIST_MAX UINT16_MAX

struct Edge {
    int id1;
    int id2;
    CashType cash;
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

double times[5+SEARCH_THREAD_COUNT];

namespace Color {
inline void reset() { fprintf(stderr, "\033[0m"); }
inline void red() { fprintf(stderr, "\033[1;31m"); }
inline void green() { fprintf(stderr, "\033[1;32m"); }
inline void yellow() { fprintf(stderr, "\033[1;33m"); }
inline void blue() { fprintf(stderr, "\033[1;34m"); }
inline void magenta() { fprintf(stderr, "\033[1;35m"); }
inline void cyan() { fprintf(stderr, "\033[1;36m"); }
inline void orange() { fprintf(stderr, "\033[38;5;214m"); }
inline void newline() { fprintf(stderr, "\n"); }
}

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
        if (val != 0) {
            inputs[sz1].cash = CashType(val);
            inputs[sz2].cash = CashType(val);
            sz1 += 2;
        }
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
    times[0] = std::chrono::duration<double>(et-st).count() * 1000.0;
#endif
}

int vtxCnt, bagCnt;
pair<int,CashType> shared_pool[MAX_EDGE_COUNT*2];
uint32_t outedge_pool[MAX_EDGE_COUNT], inedge_pool[MAX_EDGE_COUNT];
int outedge_info[MAX_VTX_COUNT][2], inedge_info[MAX_VTX_COUNT][2], outedge_info_[MAX_VTX_COUNT][2], inedge_info_[MAX_VTX_COUNT][2];
signed char idComma[MAX_VTX_COUNT][16], idComma_[MAX_VTX_COUNT][16];
int idLen[MAX_VTX_COUNT], idLen_[MAX_VTX_COUNT];
int prev_beg[MAX_VTX_COUNT];
int topo_sort[MAX_VTX_COUNT];
int to_topo_idx[MAX_VTX_COUNT];
bool topo_vis[MAX_VTX_COUNT];
bool head_record[MAX_VTX_COUNT];
bool key_node[MAX_VTX_COUNT];
int bag[MAX_VTX_COUNT];

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
            if (input[j].isInedge) ++inedge_info[vtx_cnt][1];
            else ++outedge_info[vtx_cnt][1];
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
                num[0] += inedge_info[i][1];
                num[1] += outedge_info[i][1];
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
            outedge_info[i][0] = out_offset;
            out_offset += outedge_info[i][1];
            inedge_info[i][0] = in_offset;
            in_offset += inedge_info[i][1];
            sz1 = 0;
            sz2 = 0;
            j = input_beg_;
            while (j < input_beg[i+1]) {
                if (input[j].isInedge) {
                    int id = inedge_info[i][0] + sz1;
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    shared_pool[id].first = int(idtoidx[idtoidx_beg[high]+low]);
                    shared_pool[id].second = CashType(input[j++].cash);
                    ++sz1;
                } else {
                    int id = outedge_info[i][0] + sz2;
                    int high = input[j].id2 >> 6;
                    char low = input[j].id2 & 63;
                    shared_pool[id].first = int(idtoidx[idtoidx_beg[high]+low]);
                    shared_pool[id].second = CashType(input[j++].cash);
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
    auto transfer_func = [&](PretreatInfo *infos) {
        for (int i = infos->l, i_end = infos->r, len_i = 0; i < i_end; ++i) {
            len_i = sprintf((char*)idComma[i], "%u", idxtoid[i]);
            idComma[i][len_i++] = ',';
            idLen[i] = len_i;
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
    queue<int> q;
    int topo_sz = 0;
    int cnt = 0;
    for (; cnt < vtxCnt; ++cnt) {
        if (inedge_info[cnt][1] == 0) {
            topo_vis[cnt] = true;
            q.push(cnt);
            ++cnt;
            break;
        }
    }
    if (q.empty()) {
        topo_vis[0] = true;
        q.push(0);
    }
    while (topo_sz < vtxCnt) {
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            topo_sort[topo_sz++] = cur;
            for (auto it = shared_pool+outedge_info[cur][0], it_end = it+outedge_info[cur][1]; it != it_end; ++it) {
                if (topo_vis[it->first]) continue;
                topo_vis[it->first] = true;
                q.push(it->first);
            }
        }
        for (; cnt < vtxCnt; ++cnt) {
            if (topo_vis[cnt] || inedge_info[cnt][1] != 0) continue;
            topo_vis[cnt] = true;
            q.push(cnt);
            ++cnt;
            break;
        }
        if (q.empty()) {
            for (int i = 0; i < vtxCnt; ++i) {
                if (!topo_vis[i]) {
                    topo_vis[i] = true;
                    q.push(i);
                    break;
                }
            }
        }
    }
    for (int i = 0; i < vtxCnt; ++i) to_topo_idx[topo_sort[i]] = i;
    for (int i = 0, in_pos = 0, out_pos = 0; i < vtxCnt; ++i) {
        int cur = topo_sort[i];
        outedge_info_[i][0] = out_pos;
        outedge_info_[i][1] = outedge_info[cur][1];
        for (int k = 0; k < outedge_info[cur][1]; ++k) {
            outedge_pool[out_pos++] = (uint32_t(to_topo_idx[shared_pool[outedge_info[cur][0]+k].first]) << 8) + CashType(shared_pool[outedge_info[cur][0]+k].second);
        }
        inedge_info_[i][0] = in_pos;
        prev_beg[i] = in_pos;
        inedge_info_[i][1] = inedge_info[cur][1];
        for (int k = 0; k < inedge_info[cur][1]; ++k) {
            inedge_pool[in_pos++] = (uint32_t(to_topo_idx[shared_pool[inedge_info[cur][0]+k].first]) << 8) + CashType(shared_pool[inedge_info[cur][0]+k].second);
        }
        memcpy(idComma_[i], idComma[cur], idLen[cur]);
        idLen_[i] = idLen[cur];
    }
    for (int i = 0; i < vtxCnt; ++i) {
        if (outedge_info_[i][1] == 1 && inedge_info_[i][1] == 0) {
            head_record[i] = true;
            q.push(i);
        }
    }
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        int next = outedge_pool[outedge_info_[cur][0]] >> 8;
        if (outedge_info_[next][1] == 1 && !head_record[next]) {
            for (int i = 0, i_end = inedge_info_[next][1]; i < i_end; ++i) {
                int prev = inedge_pool[inedge_info_[next][0]+i] >> 8;
                if (!head_record[prev]) break;
                if (i == i_end-1) {
                    head_record[next] = true;
                    q.push(next);
                }
            }
        }
    }
    for (int i = 0; i < vtxCnt; ++i) {
        if (head_record[i]) continue;
        for (int j = 0, j_end = inedge_info_[i][1]; j < j_end; ++j) {
            int prev = inedge_pool[inedge_info_[i][0]+j] >> 8;
            if (head_record[prev]) {
                key_node[i] = true;
            }
        }
    }
    for (int i = 0; i < vtxCnt; ++i) if (!head_record[i] && (outedge_info_[i][1] != 0 || key_node[i])) bag[bagCnt++] = i;
    shmdt(inputs);
    shmctl(shmidInputs, IPC_RMID, NULL);
    free(input_beg);
    free(idxtoid);
    free(idtoidx_beg);
    free(input);
    free(input_tmp);
    free(idtoidx);
#if TEST
    auto et = std::chrono::steady_clock::now();
    times[1] = std::chrono::duration<double>(et-st).count() * 1000.0;
#endif
}

bool visit[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
int visit_idx[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
int path_cnt[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
int prev_info[SEARCH_THREAD_COUNT][MAX_VTX_COUNT][PREV_CNT];
int shared_prev[SEARCH_THREAD_COUNT][MAX_EDGE_COUNT];
int16_t prev_sz[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
DistType dist[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
double delta[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
pair<int,double> res[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
pair<DistType,int> heap_vec[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
atomic<int> vtx_cnt_atomic(0);

inline void push(pair<DistType,int> *heap_vec_, int &heap_vec_sz, const pair<DistType,int> &&n) {
    heap_vec_[++heap_vec_sz] = n;
    int son = heap_vec_sz;
    int son_shift_1 = son >> 1;
    while (son_shift_1) {
        if (heap_vec_[son_shift_1].first > heap_vec_[son].first) {
            swap(heap_vec_[son_shift_1], heap_vec_[son]);
            son = son_shift_1;
            son_shift_1 >>= 1;
        } else break;
    }
}
inline void pop(pair<DistType,int> *heap_vec_, int &heap_vec_sz, pair<DistType,int> &tp) {
    tp = heap_vec_[1];
    swap(heap_vec_[1], heap_vec_[heap_vec_sz--]);
    int pa = 1, son = 2;
    while (son <= heap_vec_sz) {
        if (heap_vec_[son].first > heap_vec_[son+1].first && son < heap_vec_sz) ++son;
        if (heap_vec_[son].first > heap_vec_[pa].first) break;
        swap(heap_vec_[son], heap_vec_[pa]);
        pa = son;
        son = pa << 1;
    }
}

int multiple[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
int head_node[SEARCH_THREAD_COUNT][MAX_VTX_COUNT];
int head_node_sz[SEARCH_THREAD_COUNT];

int dfs(int tid, int cur, int depth) {
    int pre_node_cnt = 0;
    head_node[tid][head_node_sz[tid]++] = cur;
    for (int i = 0, i_end = inedge_info_[cur][1]; i < i_end; ++i) {
        int pre = inedge_pool[inedge_info_[cur][0]+i] >> 8;
        if (!head_record[pre]) continue;
        ++pre_node_cnt;
        delta[tid][pre] = 1.0 + delta[tid][cur];
        pre_node_cnt += dfs(tid, pre, depth+1);
    }
    multiple[tid][cur] = pre_node_cnt;
    return pre_node_cnt;
}

const size_t max_length = 1 << 16;
const size_t max_bucket_size = 1 << 16;
const size_t magical_heap_size = max_length * max_bucket_size;

struct magical_heap {
    char *region;
    int p[max_length];
    uint16_t cur;
    uint16_t term;
    magical_heap() {
        region = (decltype(region))mmap64(NULL, magical_heap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        memset(p, 0, sizeof(p));
        cur = 0;
        term = 0;
    }
    inline void push(const DistType &x, const int &item) {
        ((int *)(region + max_bucket_size * x))[p[x]++] = item;
        term = std::max((uint16_t)x, term);
    }
    ~magical_heap() { munmap(region, magical_heap_size); }
};

void search_thread(int tid) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    int heap_vec_sz = 0;
    auto visit_ = visit[tid];
    auto dist_ = dist[tid];
    auto prev_ = prev_info[tid];
    auto delta_ = delta[tid];
    auto res_ = res[tid];
    auto heap_vec_ = heap_vec[tid];
    auto prev_sz_ = prev_sz[tid];
    auto path_cnt_ = path_cnt[tid];
    auto shared_prev_ = shared_prev[tid];
    auto visit_idx_ = visit_idx[tid];
    auto multiple_ = multiple[tid];
    auto headn_node = head_node[tid];
    memset(dist_, 0xff, sizeof(DistType)*vtxCnt);
    pair<DistType,int> cur;
    int cur_idx, next_idx, w, visit_sz = 0;
    DistType cur_cash, old_dist, new_dist;
    int *prev_w, *prev_w_;
    magical_heap heap;
    for (int j = 0; j < vtxCnt; ++j) res_[j].first = j;
    for (int k = 0; k < bagCnt; ++k) {
        k = vtx_cnt_atomic++;
    //    while (!vtx_cnt_atomic.compare_exchange_weak(k, k+1, std::memory_order_relaxed));
        if (k >= bagCnt) break;
#if TEST
        cout << "thread " << tid+1 << " : " << k << " / " << bagCnt << endl;
#endif
        int idxi = bag[k];
        visit_sz = 0;
        dist_[idxi] = 0;
        path_cnt_[idxi] = 1;
        heap_vec_sz = 0;
        heap.cur = 0;
        heap.term = 0;
        heap.push(0,idxi);
        while (1) {
            if (heap.cur <= heap.term) {
                if (heap.p[heap.cur] == 0) {
                    ++heap.cur;
                    continue;
                }
                for (int i = 0; i < heap.p[heap.cur]; ++i) {
                    cur_cash = heap.cur;
                    cur_idx = ((int *)(heap.region + max_bucket_size * cur_cash))[i];
                    if (visit_[cur_idx]) continue;
                    visit_[cur_idx] = true;
                    visit_idx_[visit_sz++] = cur_idx;
                    auto it = outedge_pool+outedge_info_[cur_idx][0], it_end = it+outedge_info_[cur_idx][1];
                    while (it != it_end) {
                        next_idx = (*it) >> 8;
                        old_dist = dist_[next_idx];
                        new_dist = DistType((*it) & 0xff) + cur_cash;
                        ++it;
                        if (new_dist > old_dist) continue;
                        if (new_dist == old_dist) {
                            path_cnt_[next_idx] += path_cnt_[cur_idx];
                            if (prev_sz_[next_idx] < PREV_CNT) prev_[next_idx][prev_sz_[next_idx]++] = cur_idx;
                            else shared_prev_[prev_beg[next_idx]+prev_sz_[next_idx]++] = cur_idx;
                        } else {
                            path_cnt_[next_idx] = path_cnt_[cur_idx];
                            prev_[next_idx][0] = cur_idx;
                            prev_sz_[next_idx] = 1;
                            dist_[next_idx] = new_dist;
                            if (new_dist < max_length) heap.push(new_dist,next_idx);
                            else push(heap_vec_, heap_vec_sz, pair<DistType,int>(new_dist,next_idx));
                        }
                    }
                }
                heap.p[heap.cur] = 0;
                ++heap.cur;
            } else {
                if (heap_vec_sz) {
                    pop(heap_vec_, heap_vec_sz, cur);
                    cur_idx = cur.second;
                    cur_cash = cur.first;
                } else break;
                if (visit_[cur_idx]) continue;
                visit_[cur_idx] = true;
                visit_idx_[visit_sz++] = cur_idx;
                auto it = outedge_pool+outedge_info_[cur_idx][0], it_end = it+outedge_info_[cur_idx][1];
                while (it != it_end) {
                    next_idx = (*it) >> 8;
                    old_dist = dist_[next_idx];
                    new_dist = DistType((*it) & 0xff) + cur_cash;
                    ++it;
                    if (new_dist > old_dist) continue;
                    if (new_dist == old_dist) {
                        path_cnt_[next_idx] += path_cnt_[cur_idx];
                        if (prev_sz_[next_idx] < PREV_CNT) prev_[next_idx][prev_sz_[next_idx]++] = cur_idx;
                        else shared_prev_[prev_beg[next_idx]+prev_sz_[next_idx]++] = cur_idx;
                    } else {
                        path_cnt_[next_idx] = path_cnt_[cur_idx];
                        prev_[next_idx][0] = cur_idx;
                        prev_sz_[next_idx] = 1;
                        dist_[next_idx] = new_dist;
                        if (new_dist < max_length) heap.push(new_dist,next_idx);
                        else push(heap_vec_, heap_vec_sz, pair<DistType,int>(new_dist,next_idx));
                    }
                }
            }
        }
        for (int i = visit_sz-1; i >= 1; --i) {
            w = visit_idx_[i];
            prev_w = prev_[w];
            if (prev_sz_[w] <= PREV_CNT) {
                for (int j = 0; j < prev_sz_[w]; ++j) {
                    int v = prev_w[j];
                    delta_[v] += path_cnt_[v] * (1.0 + delta_[w]) / double(path_cnt_[w]);
                }
            } else {
                for (int j = 0; j < PREV_CNT; ++j) {
                    int v = prev_w[j];
                    delta_[v] += path_cnt_[v] * (1.0 + delta_[w]) / double(path_cnt_[w]);
                }
                prev_w_ = shared_prev_ + prev_beg[w];
                for (int j = PREV_CNT; j < prev_sz_[w]; ++j) {
                    int v = prev_w_[j];
                    delta_[v] += path_cnt_[v] * (1.0 + delta_[w]) / double(path_cnt_[w]);
                }
            }
        }
        if (!key_node[idxi]) {
            for (int i = 1; i < visit_sz; ++i) {
                w = visit_idx_[i];
                res_[w].second += delta_[w];
            }
        } else {
            int mult = dfs(tid, idxi, 0) + 1;
            for (int i = 1; i < visit_sz; ++i) multiple_[visit_idx_[i]] = mult;
            for (int i = 0, i_end = head_node_sz[tid]; i < i_end; ++i) visit_idx_[visit_sz++] = headn_node[i];
            head_node_sz[tid] = 0;
            for (int i = 1; i < visit_sz; ++i) {
                w = visit_idx_[i];
                res_[w].second += delta_[w] * multiple_[w];
            }
        }
        for (int i = visit_sz-1, idx = 0; i >= 0; --i) {
            idx = visit_idx_[i];
            delta_[idx] = 0;
            visit_[idx] = false;
            dist_[idx] = DIST_MAX;
        }
    }
#if TEST
    auto et = std::chrono::steady_clock::now();
    times[2+tid] = std::chrono::duration<double>(et-st).count();
#endif
}

void search() {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    thread search_thd[SEARCH_THREAD_COUNT];
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc] = thread(search_thread, tc);
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) search_thd[tc].join();
    int vtx_offset[SEARCH_THREAD_COUNT+1] = {0};
    for (int tc = 0, last = 0; tc < SEARCH_THREAD_COUNT; ++tc) {
        vtx_offset[tc] = last;
        last = (tc+1) * vtxCnt / SEARCH_THREAD_COUNT;
    }
    vtx_offset[SEARCH_THREAD_COUNT] = vtxCnt;
    auto func = [&vtx_offset](int tid) {
        for (int i = vtx_offset[tid], i_end = vtx_offset[tid+1]; i < i_end; ++i) {
            for (int j = 1; j < SEARCH_THREAD_COUNT; ++j) {
                res[0][i].second += res[j][i].second;
            }
        }
    };
    thread merge_thread[SEARCH_THREAD_COUNT];
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) merge_thread[tc] = thread(func, tc);
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) merge_thread[tc].join();
    sort(res[0], res[0]+vtxCnt, [](pair<int,double>& p1, pair<int,double>& p2)->bool{if (fabs(p1.second - p2.second)>0.0001) return p1.second > p2.second;else return topo_sort[p1.first] < topo_sort[p2.first];});
#if TEST
    auto et = std::chrono::steady_clock::now();
    times[2+SEARCH_THREAD_COUNT] = std::chrono::duration<double>(et-st).count();
#endif
}

void store(const string &resultFile) {
#if TEST
    auto st = std::chrono::steady_clock::now();
#endif
    FILE *fptr;
    fptr = fopen(resultFile.c_str(), "w");
    char dot[1] = {'.'}, LF[1] = {'\n'}, zero[3] = {'0','0','0'};
    int output_sz = vtxCnt < OUTPUT_LIMIT ? vtxCnt : OUTPUT_LIMIT;
    for (int i = 0; i < output_sz; ++i) {
        fwrite(idComma_[res[0][i].first], idLen_[res[0][i].first], 1, fptr);
        uint64_t ival = uint64_t(round(res[0][i].second*1000));
        string high_str = to_string(ival/1000);
        string low_str = to_string(ival%1000);
        fwrite(high_str.c_str(), high_str.size(), 1, fptr);
        fwrite(dot, 1, 1, fptr);
        fwrite(zero, 3-low_str.size(), 1, fptr);
        fwrite(low_str.c_str(), low_str.size(), 1, fptr);
        fwrite(LF, 1, 1, fptr);
    }
    fclose(fptr);
#if TEST
    auto et = std::chrono::steady_clock::now();
    times[3+SEARCH_THREAD_COUNT] = std::chrono::duration<double>(et-st).count() * 1000.0;
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
    times[4+SEARCH_THREAD_COUNT] = std::chrono::duration<double>(et-st).count();
    Color::reset();
    Color::yellow();
    fprintf(stderr, "-------------------------\n");
    fprintf(stderr, "@ the load  : %f ms.\n", times[0]);
    fprintf(stderr, "-------------------------\n");
    fprintf(stderr, "@ the build : %f ms.\n", times[1]);
    fprintf(stderr, "-------------------------\n");
    Color::cyan();
    for (int tc = 0; tc < SEARCH_THREAD_COUNT; ++tc) fprintf(stderr, "@ the thd %d : %f s.\n", tc+1, times[2+tc]);
    Color::green();
    fprintf(stderr, "-------------------------\n");
    fprintf(stderr, "@ the solve : %f s.\n", times[2+SEARCH_THREAD_COUNT]);
    fprintf(stderr, "-------------------------\n");
    fprintf(stderr, "@ the store : %f ms.\n", times[3+SEARCH_THREAD_COUNT]);
    fprintf(stderr, "-------------------------\n");
    fprintf(stderr, "@ the total : %f s.\n", times[4+SEARCH_THREAD_COUNT]);
    fprintf(stderr, "-------------------------\n");
    Color::reset();
#endif
    exit(0);
}
