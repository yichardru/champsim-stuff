#include "d_mockingjay.h"

#include <algorithm>
#include <cmath>

#include "champsim.h"

d_mockingjay::d_mockingjay(CACHE* cache) : replacement(cache), NUM_SET(static_cast<long>(cache->NUM_SET)), NUM_WAY(static_cast<long>(cache->NUM_WAY))
{
  // Compute log2 values
  LOG2_LLC_SET = __builtin_ctz(static_cast<unsigned>(NUM_SET));
  const int log2_num_way = __builtin_ctz(static_cast<unsigned>(NUM_WAY));
  LOG2_LLC_SIZE = LOG2_LLC_SET + log2_num_way + LOG2_BLOCK_SIZE;
  LOG2_SAMPLED_SETS = LOG2_LLC_SIZE - 16;

  INF_RD = static_cast<int>(NUM_WAY) * HISTORY - 1;
  INF_ETR = (static_cast<int>(NUM_WAY) * HISTORY / GRANULARITY) - 1;
  MAX_RD = INF_RD - 22;

  SAMPLED_CACHE_TAG_BITS = 31 - LOG2_LLC_SIZE;
  PC_SIGNATURE_BITS = LOG2_LLC_SIZE - 10;

  FLEXMIN_PENALTY = 2.0 - std::log2(static_cast<double>(NUM_CPUS)) / 4.0;
  D_LINES_PER_SLICE = NUM_SET * NUM_WAY;

  // Allocate state vectors
  etr.assign(NUM_SET * NUM_WAY, 0);
  etr_clock.assign(NUM_SET, GRANULARITY);
  current_timestamp.assign(NUM_SET, 0);
  mpka_counter.assign(NUM_SET, 0);
  is_d_sampled.assign(NUM_SET, false);
}

/**
 * Helpers
 */
void d_mockingjay::allocate_sampled_cache_for_set(uint32_t s)
{
  // Allocate sampled cache entris for s if not already done so
  int modifier = 1 << LOG2_LLC_SET;
  int limit = 1 << LOG2_SAMPLED_CACHE_SETS;
  for (int i = 0; i < limit; i++) {
    uint32_t idx = s + modifier * i;
    if (!sampled_cache.count(idx)) {
      sampled_cache[idx] = new SampledCacheLine[SAMPLED_CACHE_WAYS]();
    }
  }
}

void d_mockingjay::d_select_sampled_sets()
{
  // Select sampled sets based on MPKA counter values
  std::vector<std::pair<int, int>> ranked;
  ranked.reserve(NUM_SET);
  for (int i = 0; i < NUM_SET; i++) {
    ranked.push_back({mpka_counter[i], i});
  }

  // If MPKA values don't vary much, fall back to uniform sampling
  int max_val = ranked[0].first, min_val = ranked[0].first;
  for (auto& p : ranked) {
    max_val = std::max(max_val, p.first);
    min_val = std::min(min_val, p.first);
  }

  if (max_val - min_val < D_UNIFORM_THRESHOLD) {
    d_uniform_mode = true;
    std::fill(is_d_sampled.begin(), is_d_sampled.end(), false);
    return;
  }

  // Otherwise, select sets with highest MPKA values
  d_uniform_mode = false;
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
              return a.first > b.first;
            });
  std::fill(is_d_sampled.begin(), is_d_sampled.end(), false);
  for (int i = 0; i < D_LINES_PER_SLICE && i < NUM_SET; i++) {
    is_d_sampled[ranked[i].second] = true;
  }
}

void d_mockingjay::d_reset_counters()
{
  std::fill(mpka_counter.begin(), mpka_counter.end(), 0);
}

bool d_mockingjay::is_sampled_set(int set)
{
  if (d_uniform_mode) {
    int mask_length = LOG2_LLC_SET - LOG2_SAMPLED_SETS;
    int mask = (1 << mask_length) - 1;
    return (set & mask) == ((set >> (LOG2_LLC_SET - mask_length)) & mask);
  }
  return is_d_sampled[set];
}

uint64_t d_mockingjay::CRC_HASH(uint64_t _blockAddress)
{
  static const unsigned long long crcPolynomial = 3988292384ULL;
  unsigned long long _returnVal = _blockAddress;
  for (unsigned int i = 0; i < 3; i++)
    _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
  return _returnVal;
}

uint64_t d_mockingjay::get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core)
{
  if (NUM_CPUS == 1) {
    pc = pc << 1;
    if (hit) 
      pc = pc | 1;
    pc = pc << 1;
    if (prefetch) 
      pc = pc | 1;
    pc = CRC_HASH(pc);
    pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
  } else {
    pc = pc << 1;
    if (prefetch) 
      pc = pc | 1;
    pc = pc << 2;
    pc = pc | core;
    pc = CRC_HASH(pc);
    pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
  }
  return pc;
}

uint32_t d_mockingjay::get_sampled_cache_index(uint64_t full_addr)
{
  full_addr = full_addr >> LOG2_BLOCK_SIZE;
  full_addr = (full_addr << (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET))) >> (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET));
  return full_addr;
}

uint64_t d_mockingjay::get_sampled_cache_tag(uint64_t x)
{
  x >>= LOG2_LLC_SET + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS;
  x = (x << (64 - SAMPLED_CACHE_TAG_BITS)) >> (64 - SAMPLED_CACHE_TAG_BITS);
  return x;
}

int d_mockingjay::search_sampled_cache(uint64_t blockAddress, uint32_t set)
{
  SampledCacheLine* sampled_set = sampled_cache[set];
  for (int way = 0; way < SAMPLED_CACHE_WAYS; way++) {
    if (sampled_set[way].valid && (sampled_set[way].tag == blockAddress))
      return way;
  }
  return -1;
}

void d_mockingjay::detrain(uint32_t set, int way)
{
  SampledCacheLine temp = sampled_cache[set][way];
  if (!temp.valid) return;
  if (rdp.count(temp.signature)) {
    rdp[temp.signature] = std::min(rdp[temp.signature] + 1, INF_RD);
  } else {
    rdp[temp.signature] = INF_RD;
  }
  sampled_cache[set][way].valid = false;
}

int d_mockingjay::temporal_difference(int init, int sample)
{
  if (sample > init) {
    int diff = sample - init;
    diff = diff * TEMP_DIFFERENCE;
    diff = std::max(1, diff);
    return std::min(init + diff, INF_RD);
  } else if (sample < init) {
    int diff = init - sample;
    diff = diff * TEMP_DIFFERENCE;
    diff = std::max(1, diff);
    return std::max(init - diff, 0);
  } else {
    return init;
  }
}

int d_mockingjay::increment_timestamp(int input)
{
  input++;
  input = input % (1 << TIMESTAMP_BITS);
  return input;
}

int d_mockingjay::time_elapsed(int global, int local)
{
  if (global >= local) 
    return global - local;
  global = global + (1 << TIMESTAMP_BITS);
  return global - local;
}

/**
 * Public interface
 */
void d_mockingjay::initialize_replacement()
{
  // put your own initialization code here
  for (int i = 0; i < NUM_SET; i++) {
    etr_clock[i] = GRANULARITY;
    current_timestamp[i] = 0;
  }
  std::fill(mpka_counter.begin(), mpka_counter.end(), 0);
  std::fill(is_d_sampled.begin(), is_d_sampled.end(), false);
  d_access_count = 0;
  d_uniform_mode = true;

  for (uint32_t set = 0; set < (uint32_t)NUM_SET; set++) {
    if (is_sampled_set(set)) {
      allocate_sampled_cache_for_set(set);
    }
  }
}

long d_mockingjay::find_victim(uint32_t cpu, uint64_t /*instr_id*/, long set, const champsim::cache_block* current_set, champsim::address pc, champsim::address /*full_addr*/, access_type type)
{
  // Check for invalid lines first
  for (int way = 0; way < NUM_WAY; way++) {
    if (current_set[way].valid == false)
      return way;
  }

  // Find maximum ETR value
  int max_etr = 0;
  int victim_way = 0;
  for (int way = 0; way < NUM_WAY; way++) {
    int e = get_etr(set, way);
    if (std::abs(e) > max_etr || (std::abs(e) == max_etr && e < 0)) {
      max_etr = std::abs(e);
      victim_way = way;
    }
  }

  // Bypass decision
  uint64_t pc_signature = get_pc_signature(pc.to<uint64_t>(), false, type == access_type::PREFETCH, cpu);
  if (type != access_type::WRITE && rdp.count(pc_signature) &&
      (rdp[pc_signature] > MAX_RD || rdp[pc_signature] / GRANULARITY > max_etr)) {
    return NUM_WAY;
  }

  return victim_way;
}

void d_mockingjay::update_replacement_state(uint32_t cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address /*victim_addr*/, access_type type, uint8_t hit)
{
  uint64_t pc = ip.to<uint64_t>();

  if (type == access_type::WRITE) {
    if (!hit) 
      get_etr(set, way) = -INF_ETR;
    return;
  }

  // Drishti MPKA tracking
  if (!hit) {
    mpka_counter[set] = std::min(mpka_counter[set] + 1, MPKA_COUNTER_MAX);
  } else {
    mpka_counter[set] = std::max(mpka_counter[set] - 1, 0);
  }

  d_access_count++;

  // Periodically select sampled sets based on MPKA values and reset counters
  if (d_access_count == D_MONITOR_INTERVAL) {
    d_select_sampled_sets();
    for (uint32_t s = 0; s < (uint32_t)NUM_SET; s++) {
      if (is_d_sampled[s]) {
        allocate_sampled_cache_for_set(s);
      }
    }
  }

  if (d_access_count >= D_RESELECT_INTERVAL) {
    d_reset_counters();
    d_select_sampled_sets();
    d_access_count = 0; 
    for (uint32_t s = 0; s < (uint32_t)NUM_SET; s++) {
      if (is_d_sampled[s]) {
        allocate_sampled_cache_for_set(s);
      }
    }
  }

  // Compute PC signature for this access
  pc = get_pc_signature(pc, hit, type == access_type::PREFETCH, cpu);

  // Update sampled cache
  if (is_sampled_set(set)) {
    uint32_t sampled_cache_index = get_sampled_cache_index(full_addr.to<uint64_t>());
    uint64_t sampled_cache_tag   = get_sampled_cache_tag(full_addr.to<uint64_t>());
    int sampled_cache_way = search_sampled_cache(sampled_cache_tag, sampled_cache_index);

    if (sampled_cache_way > -1) {
      uint64_t last_signature = sampled_cache[sampled_cache_index][sampled_cache_way].signature;
      uint64_t last_timestamp = sampled_cache[sampled_cache_index][sampled_cache_way].timestamp;
      int sample = time_elapsed(current_timestamp[set], last_timestamp);
      if (sample <= INF_RD) {
        if (type == access_type::PREFETCH)
          sample = sample * FLEXMIN_PENALTY;
        if (rdp.count(last_signature)) {
          int init = rdp[last_signature];
          rdp[last_signature] = temporal_difference(init, sample);
        } else {
          rdp[last_signature] = sample;
        }
        sampled_cache[sampled_cache_index][sampled_cache_way].valid = false;
      }
    }

    // Find a victim in the sampled cache (LRU approximation)
    int lru_way = -1;
    int lru_rd  = -1;
    for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
      if (sampled_cache[sampled_cache_index][w].valid == false) {
        lru_way = w;
        lru_rd  = INF_RD + 1;
        continue;
      }

      uint64_t last_timestamp = sampled_cache[sampled_cache_index][w].timestamp;
      int sample = time_elapsed(current_timestamp[set], last_timestamp);
      if (sample > INF_RD) {
        lru_way = w;
        lru_rd  = INF_RD + 1;
        detrain(sampled_cache_index, w);
      } else if (sample > lru_rd) {
        lru_way = w;
        lru_rd  = sample;
      }
    }
    detrain(sampled_cache_index, lru_way);

    // Insert new entry
    for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
      if (sampled_cache[sampled_cache_index][w].valid == false) {
        sampled_cache[sampled_cache_index][w].valid     = true;
        sampled_cache[sampled_cache_index][w].signature = pc;
        sampled_cache[sampled_cache_index][w].tag       = sampled_cache_tag;
        sampled_cache[sampled_cache_index][w].timestamp = current_timestamp[set];
        break;
      }
    }

    current_timestamp[set] = increment_timestamp(current_timestamp[set]);
  }

  // Update ETR for all ways in set
  if (etr_clock[set] == GRANULARITY) {
    for (int w = 0; w < NUM_WAY; w++) {
      if (w != way && std::abs(get_etr(set, w)) < INF_ETR)
        get_etr(set, w)--;
    }
    etr_clock[set] = 0;
  }
  etr_clock[set]++;

  // Set ETR for the accessed way based on predicted reuse distance
  if (way < NUM_WAY) {
    if (!rdp.count(pc)) {
      get_etr(set, way) = (NUM_CPUS == 1) ? 0 : INF_ETR;
    } else {
      if (rdp[pc] > MAX_RD)
        get_etr(set, way) = INF_ETR;
      else
        get_etr(set, way) = rdp[pc] / GRANULARITY;
    }
  }
}

void d_mockingjay::replacement_final_stats() {}
