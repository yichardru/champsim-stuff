#ifndef REPLACEMENT_D_MOCKINGJAY_H
#define REPLACEMENT_D_MOCKINGJAY_H

#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

struct d_mockingjay : public champsim::modules::replacement {
private:
  static constexpr int HISTORY = 8;
  static constexpr int GRANULARITY = 8;
  static constexpr int SAMPLED_CACHE_WAYS = 5;
  static constexpr int LOG2_SAMPLED_CACHE_SETS = 4;
  static constexpr int TIMESTAMP_BITS = 8;
  static constexpr double TEMP_DIFFERENCE = 1.0 / 16.0;
  static constexpr int D_SAMPLED_SETS_PER_SLICE = 16;
  static constexpr int D_MONITOR_INTERVAL = 32768;
  static constexpr int D_RESELECT_INTERVAL = 131072;
  static constexpr int D_UNIFORM_THRESHOLD = 100;
  static constexpr int MPKA_COUNTER_MAX = 255;

  long NUM_SET, NUM_WAY;
  int LOG2_LLC_SET, LOG2_LLC_SIZE, LOG2_SAMPLED_SETS;
  int INF_RD, INF_ETR, MAX_RD;
  int SAMPLED_CACHE_TAG_BITS, PC_SIGNATURE_BITS;
  double FLEXMIN_PENALTY;
  long D_LINES_PER_SLICE;

  std::vector<int> etr;
  std::vector<int> etr_clock;
  std::vector<int> current_timestamp;
  std::vector<int> mpka_counter;
  std::vector<bool> is_d_sampled;

  std::unordered_map<uint32_t, int> rdp;

  struct SampledCacheLine {
    bool valid = false;
    uint64_t tag = 0;
    uint64_t signature = 0;
    int timestamp = 0;
  };
  std::unordered_map<uint32_t, SampledCacheLine*> sampled_cache;

  int d_access_count = 0;
  bool d_uniform_mode = true;

  int& get_etr(long set, long way) { return etr[set * NUM_WAY + way]; }

  void d_select_sampled_sets();
  void d_reset_counters();
  bool is_sampled_set(int set);
  void allocate_sampled_cache_for_set(uint32_t s);
  uint64_t CRC_HASH(uint64_t addr);
  uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core);
  uint32_t get_sampled_cache_index(uint64_t full_addr);
  uint64_t get_sampled_cache_tag(uint64_t x);
  int search_sampled_cache(uint64_t blockAddress, uint32_t set);
  void detrain(uint32_t set, int way);
  int temporal_difference(int init, int sample);
  int increment_timestamp(int input);
  int time_elapsed(int global, int local);

public:
  explicit d_mockingjay(CACHE* cache);
  void initialize_replacement();
  long find_victim(uint32_t cpu, uint64_t instr_id, long set,
                   const champsim::cache_block* current_set,
                   champsim::address pc, champsim::address full_addr,
                   access_type type);
  void update_replacement_state(uint32_t cpu, long set, long way,
                                champsim::address full_addr, champsim::address ip,
                                champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif // REPLACEMENT_D_MOCKINGJAY_H
