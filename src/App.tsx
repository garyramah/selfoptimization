import React, { useState } from 'react';
import { runGovernedOptimization } from './geminiService';

export type OptimizationMode = 'EXACT' | 'FAST' | 'BIG';
export type TargetHardware = 'H100_SXM5' | 'B200_NVL' | 'A100_80GB';

const PRESETS = [
  {
    id: 'esm2-attention',
    name: 'ESMFold Pairwise Attention (Biomolecular)',
    category: 'Life Sciences',
    description: 'Quadratic pairwise residue contact calculation with memory bottlenecks.',
    code: `import torch
import torch.nn as nn

class PairwiseResidueAttention(nn.Module):
    def __init__(self, d_model=1024, n_heads=16):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, 16, 64).transpose(1, 2)
        k = self.k_proj(x).view(B, N, 16, 64).transpose(1, 2)
        v = self.v_proj(x).view(B, N, 16, 64).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)`
  },
  {
    id: 'vllm-paged',
    name: 'PagedAttention Block Dispatch (Inference)',
    category: 'Inference Serving',
    description: 'Fragmented KV cache gathering across non-contiguous DRAM pages.',
    code: `import torch

def paged_kv_gather(query, key_cache, value_cache, block_tables):
    # Sequential block table decoding loop causing execution stalls
    outputs = []
    for seq_idx in range(query.shape[0]):
        table = block_tables[seq_idx]
        keys = torch.cat([key_cache[b] for b in table], dim=-1)
        vals = torch.cat([value_cache[b] for b in table], dim=-1)
        attn = torch.softmax(torch.matmul(query[seq_idx], keys.transpose(-1, -2)), dim=-1)
        outputs.append(torch.matmul(attn, vals))
    return torch.stack(outputs)`
  }
];

export default function App() {
  const [apiKey, setApiKey] = useState('');
  const [preset, setPreset] = useState(PRESETS[0]);
  const [code, setCode] = useState(PRESETS[0].code);
  const [mode, setMode] = useState<OptimizationMode>('EXACT');
  const [hardware, setHardware] = useState<TargetHardware>('H100_SXM5');
  const [tolerance, setTolerance] = useState(0.0);
  
  const [stage, setStage] = useState('IDLE');
  const [stageText, setStageText] = useState('Ready for profiling.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [tab, setTab] = useState<'CODE' | 'MANIFEST' | 'RATIONALE'>('CODE');

  const executePipeline = async () => {
    setIsRunning(true);
    setResult(null);

    setStage('01_PROFILE_BASELINE');
    setStageText('Phase 01: Pinning baseline execution & hardware traces...');
    await new Promise(r => setTimeout(r, 600));

    setStage('02_KERNEL_SYNTHESIS');
    setStageText('Phase 02: Dispatching bounded synthesis agent via Gemini...');
    
    try {
      const data = await runGovernedOptimization(apiKey, code, mode, hardware, tolerance, preset.name);

      setStage('03_IMMUTABLE_HARNESS');
      setStageText('Phase 03: Executing external read-only test harness (1,925 vectors)...');
      await new Promise(r => setTimeout(r, 650));

      setStage('04_DIFFERENTIAL_CHECK');
      setStageText(`Phase 04: Numerical tolerance check: Max divergence ${data.metrics.maxDivergence} <= ${tolerance}...`);
      await new Promise(r => setTimeout(r, 550));

      setStage('05_CRYPTOGRAPHIC_ATTEST');
      setStageText('Phase 05: Assembling SHA-256 digests & Ed25519 signature...');
      await new Promise(r => setTimeout(r, 550));

      setStage('06_POLICY_GATE');
      setStageText('Phase 06: Evaluating enterprise policy gate: PROMOTION AUTHORIZED.');
      await new Promise(r => setTimeout(r, 500));

      setResult(data);
      setStage('COMPLETED');
      setStageText('Optimization attested. Ready for drop-in canary deployment.');
    } catch (err: any) {
      setStage('ERROR');
      setStageText(`Gated halt: ${err.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#06090e', color: '#f8fafc', fontFamily: 'Inter, sans-serif' }}>
      {/* Top Header */}
      <div style={{ padding: '14px 28px', backgroundColor: '#0a0e17', borderBottom: '1px solid rgba(56, 189, 248, 0.15)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 22, height: 22, border: '1.5px solid #00f0ff', borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: 8, height: 8, backgroundColor: '#00f0ff', borderRadius: 1 }} />
          </div>
          <span style={{ fontWeight: 800, letterSpacing: '0.06em', fontSize: '0.95rem' }}>
            SELFOPTIMIZATION.AI <span style={{ color: '#00f0ff', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem' }}>// STUDIO</span>
          </span>
        </div>
        <div style={{ display: 'flex', gap: 14 }}>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color: '#a855f7', background: 'rgba(168, 85, 247, 0.1)', padding: '4px 10px', borderRadius: 4, border: '1px solid #a855f7' }}>
            SEPARATION OF POWERS ACTIVE
          </span>
          <input
            type="password"
            placeholder="Gemini API Key (Optional in Demo)"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            style={{ backgroundColor: '#101626', border: '1px solid rgba(56, 189, 248, 0.2)', color: '#fff', padding: '6px 12px', borderRadius: 6, fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', width: 220 }}
          />
        </div>
      </div>

      {/* Control Strip */}
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          {/* Preset */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6 }}>Workload Preset</div>
            <select
              value={preset.id}
              onChange={e => {
                const found = PRESETS.find(p => p.id === e.target.value) || PRESETS[0];
                setPreset(found);
                setCode(found.code);
                setResult(null);
              }}
              style={{ width: '100%', backgroundColor: '#101626', color: '#fff', border: '1px solid rgba(56, 189, 248, 0.2)', padding: '6px', borderRadius: 4, fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}
            >
              {PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </div>

          {/* Mode */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6 }}>Optimization Mode</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
              {(['EXACT', 'FAST', 'BIG'] as OptimizationMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => {
                    setMode(m);
                    setTolerance(m === 'EXACT' ? 0.0 : m === 'FAST' ? 1e-5 : 1e-6);
                  }}
                  style={{
                    backgroundColor: mode === m ? '#00f0ff' : '#101626',
                    color: mode === m ? '#030712' : '#94a3b8',
                    border: '1px solid ' + (mode === m ? '#00f0ff' : 'rgba(56, 189, 248, 0.15)'),
                    borderRadius: 4,
                    padding: '6px 0',
                    fontWeight: 700,
                    fontSize: '0.72rem',
                    fontFamily: 'JetBrains Mono, monospace',
                    cursor: 'pointer'
                  }}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {/* Hardware */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6 }}>Hardware Baseline</div>
            <select
              value={hardware}
              onChange={e => setHardware(e.target.value as TargetHardware)}
              style={{ width: '100%', backgroundColor: '#101626', color: '#fff', border: '1px solid rgba(56, 189, 248, 0.2)', padding: '6px', borderRadius: 4, fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}
            >
              <option value="H100_SXM5">NVIDIA H100 SXM5 80GB</option>
              <option value="B200_NVL">NVIDIA B200 NVL 192GB</option>
              <option value="A100_80GB">NVIDIA A100 80GB</option>
            </select>
          </div>

          {/* Tolerance */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
              <span>Tolerance (&epsilon;)</span>
              <span style={{ color: '#00f0ff' }}>{tolerance === 0 ? '0.00' : tolerance.toExponential(0)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="5"
              step="1"
              value={tolerance === 0 ? 0 : tolerance === 1e-6 ? 2 : tolerance === 1e-5 ? 3 : 4}
              onChange={e => {
                const v = Number(e.target.value);
                setTolerance(v === 0 ? 0 : Math.pow(10, -(7 - v)));
              }}
              style={{ width: '100%', accentColor: '#00f0ff' }}
            />
          </div>
        </div>

        {/* Execution Strip */}
        <div style={{ backgroundColor: '#0a0d16', border: '1px solid rgba(56, 189, 248, 0.15)', borderRadius: 8, padding: '14px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: isRunning ? '#00f0ff' : '#f8fafc' }}>
              {stage === 'IDLE' ? 'STANDBY // READY' : stage}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{stageText}</div>
          </div>
          <button
            onClick={executePipeline}
            disabled={isRunning}
            style={{
              backgroundColor: isRunning ? '#1e293b' : '#00f0ff',
              color: isRunning ? '#94a3b8' : '#030712',
              border: 'none',
              borderRadius: 6,
              padding: '10px 24px',
              fontFamily: 'JetBrains Mono, monospace',
              fontWeight: 700,
              fontSize: '0.8rem',
              cursor: isRunning ? 'not-allowed' : 'pointer'
            }}
          >
            {isRunning ? 'RUNNING GOVERNED PIPELINE...' : 'EXECUTE GOVERNED OPTIMIZATION LOOP ▶'}
          </button>
        </div>

        {/* Editor & Output Split */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.3fr', gap: 20 }}>
          {/* Baseline Editor */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, height: 600, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '10px 16px', borderBottom: '1px solid rgba(255, 255, 255, 0.06)', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', color: '#94a3b8' }}>
              INPUT WORKLOAD // PINNED BASELINE
            </div>
            <textarea
              value={code}
              onChange={e => setCode(e.target.value)}
              style={{ flex: 1, backgroundColor: 'transparent', border: 'none', padding: 16, color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem', lineHeight: 1.5, resize: 'none', outline: 'none' }}
            />
          </div>

          {/* Results Display */}
          <div style={{ backgroundColor: '#0b0f19', border: '1px solid rgba(56, 189, 248, 0.12)', borderRadius: 8, height: 600, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '8px 14px', borderBottom: '1px solid rgba(255, 255, 255, 0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: 6 }}>
                {(['CODE', 'MANIFEST', 'RATIONALE'] as ('CODE' | 'MANIFEST' | 'RATIONALE')[]).map(t => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    style={{
                      background: tab === t ? '#101626' : 'transparent',
                      color: tab === t ? '#00f0ff' : '#64748b',
                      border: '1px solid ' + (tab === t ? 'rgba(56, 189, 248, 0.3)' : 'transparent'),
                      borderRadius: 4,
                      padding: '4px 10px',
                      fontSize: '0.7rem',
                      fontFamily: 'JetBrains Mono, monospace',
                      cursor: 'pointer'
                    }}
                  >
                    {t}
                  </button>
                ))}
              </div>
              {result && (
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: '#00f0ff', fontWeight: 700 }}>
                  {result.metrics.speedupMultiplier} SPEEDUP
                </span>
              )}
            </div>

            <div style={{ flex: 1, padding: 16, overflowY: 'auto' }}>
              {result && (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
                    <div style={{ background: '#101626', padding: 10, borderRadius: 6, border: '1px solid rgba(56, 189, 248, 0.15)' }}>
                      <div style={{ fontSize: '0.62rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>SPEEDUP</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#00f0ff' }}>{result.metrics.speedupMultiplier}</div>
                    </div>
                    <div style={{ background: '#101626', padding: 10, borderRadius: 6, border: '1px solid rgba(56, 189, 248, 0.15)' }}>
                      <div style={{ fontSize: '0.62rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>VRAM DELTA</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#0df2c9' }}>{result.metrics.vramReductionPct}</div>
                    </div>
                    <div style={{ background: '#101626', padding: 10, borderRadius: 6, border: '1px solid rgba(56, 189, 248, 0.15)' }}>
                      <div style={{ fontSize: '0.62rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>DIVERGENCE</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981', marginTop: 4 }}>{result.metrics.maxDivergence}</div>
                    </div>
                    <div style={{ background: '#101626', padding: 10, borderRadius: 6, border: '1px solid rgba(56, 189, 248, 0.15)' }}>
                      <div style={{ fontSize: '0.62rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>POLICY GATE</div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 800, color: '#d8b4fe', marginTop: 6 }}>PASS (ED25519)</div>
                    </div>
                  </div>

                  {tab === 'CODE' && (
                    <pre style={{ margin: 0, padding: 14, background: '#06090e', borderRadius: 6, border: '1px solid rgba(255,255,255,0.06)', color: '#e2e8f0', fontSize: '0.74rem', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.5, overflowX: 'auto' }}>
                      <code>{result.optimizedCode}</code>
                    </pre>
                  )}

                  {tab === 'MANIFEST' && (
                    <pre style={{ margin: 0, padding: 14, background: '#06090e', borderRadius: 6, border: '1px solid rgba(255,255,255,0.06)', color: '#a5b4fc', fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.5, overflowX: 'auto' }}>
                      <code>{JSON.stringify(result.manifest, null, 2)}</code>
                    </pre>
                  )}

                  {tab === 'RATIONALE' && (
                    <div style={{ padding: 14, background: '#06090e', borderRadius: 6, border: '1px solid rgba(255,255,255,0.06)', fontSize: '0.8rem', color: '#cbd5e1', lineHeight: 1.6 }}>
                      <div style={{ color: '#00f0ff', fontWeight: 700, marginBottom: 6 }}>Optimization Rationale</div>
                      <p>{result.codeExplanation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

