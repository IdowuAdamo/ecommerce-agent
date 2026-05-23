"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Zap, Shield, TrendingUp, Star, ExternalLink, ChevronDown, ChevronRight, Loader2 } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────
interface PriceFairness {
  actual_price: number;
  predicted_fair_price: number;
  price_deviation_pct: number;
  verdict: string;
  fairness_score: number;
  explanation: string;
}

interface TrustScore {
  overall: number;
  seller_score: number;
  rating_authenticity: number;
  flags: string[];
}

interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  old_price?: number;
  discount_pct?: number;
  rating?: number;
  num_reviews?: number;
  seller?: string;
  product_url?: string;
  image_url?: string;
  brand?: string;
  price_fairness?: PriceFairness;
  trust_score?: TrustScore;
}

interface RankedProduct {
  product: Product;
  rank: number;
  composite_score: number;
  semantic_score: number;
  behavioral_score: number;
  price_fairness_score: number;
  trust_score_val: number;
  explanation?: string;
}

interface ExplanationCard {
  product_id: string;
  headline: string;
  reasons: string[];
  warnings: string[];
  value_verdict: string;
  nigerian_context: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  recommendations?: RankedProduct[];
  explanations?: ExplanationCard[];
  agentSteps?: string[];
  timestamp: Date;
}

// ── Utility Functions ─────────────────────────────────────────────────────────
const formatNaira = (amount: number) =>
  `₦${amount.toLocaleString("en-NG")}`;

const getVerdictColor = (verdict: string) => {
  switch (verdict) {
    case "great_deal": return "verdict-deal";
    case "fair": return "verdict-fair";
    case "slightly_overpriced": return "verdict-overpriced";
    case "overpriced": case "suspicious": return "verdict-suspicious";
    default: return "text-[var(--text-secondary)]";
  }
};

const getVerdictLabel = (verdict: string) => {
  switch (verdict) {
    case "great_deal": return "🔥 Great Deal";
    case "fair": return "✅ Fair Price";
    case "slightly_overpriced": return "⚠️ Slightly Pricey";
    case "overpriced": return "❌ Overpriced";
    case "suspicious": return "🚨 Suspicious Price";
    default: return verdict;
  }
};

const getTrustLabel = (score: number) => {
  if (score >= 0.75) return { label: "High Trust", cls: "trust-high" };
  if (score >= 0.5) return { label: "Moderate", cls: "trust-medium" };
  return { label: "Low Trust", cls: "trust-low" };
};

const EXAMPLE_QUERIES = [
  "Find me a durable laptop under ₦400k for machine learning",
  "I need a phone under ₦150k with good camera — NYSC allowance",
  "Best Samsung TV under ₦600k for my sitting room",
  "Affordable blender for Nigerian kitchen, abeg no expensive one",
  "Gaming laptop under ₦800k for heavy use",
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Components ────────────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-[var(--accent-green)] opacity-70"
          style={{ animation: `typing-bounce 1.2s ease ${i * 0.2}s infinite` }}
        />
      ))}
      <span className="text-xs text-[var(--text-muted)] ml-2">Agents reasoning...</span>
    </div>
  );
}

function AgentSteps({ steps }: { steps: string[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!steps?.length) return null;

  return (
    <div className="mt-3 rounded-lg border border-[var(--border)] overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
      >
        <span className="flex items-center gap-2">
          <Zap size={12} className="text-[var(--accent-green)]" />
          Agent Reasoning Trace ({steps.length} steps)
        </span>
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {expanded && (
        <div className="px-3 pb-3 space-y-1 border-t border-[var(--border)]">
          {steps.map((step, i) => (
            <div key={i} className="agent-step animate-fade-in" style={{ animationDelay: `${i * 50}ms` }}>
              {step}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PriceFairnessBar({ fairness }: { fairness: PriceFairness }) {
  const barWidth = Math.max(5, Math.min(100, fairness.fairness_score * 100));
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center text-xs">
        <span className="text-[var(--text-muted)]">Price Fairness</span>
        <span className={`font-medium ${getVerdictColor(fairness.verdict)}`}>
          {getVerdictLabel(fairness.verdict)}
        </span>
      </div>
      <div className="h-1.5 bg-[var(--surface)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${barWidth}%`,
            background: fairness.verdict === "great_deal" || fairness.verdict === "fair"
              ? "linear-gradient(90deg, var(--naija-green), var(--accent-green))"
              : fairness.verdict === "slightly_overpriced"
              ? "linear-gradient(90deg, var(--gold), var(--accent-orange))"
              : "linear-gradient(90deg, var(--accent-red), var(--accent-orange))",
          }}
        />
      </div>
      <div className="flex justify-between text-xs text-[var(--text-muted)]">
        <span>Actual: {formatNaira(fairness.actual_price)}</span>
        <span>Fair Est: {formatNaira(fairness.predicted_fair_price)}</span>
      </div>
    </div>
  );
}

function ProductCard({
  ranked,
  explanation,
  animDelay,
}: {
  ranked: RankedProduct;
  explanation?: ExplanationCard;
  animDelay: number;
}) {
  const p = ranked.product;
  const trust = getTrustLabel(ranked.trust_score_val);
  const [showExplanation, setShowExplanation] = useState(false);

  return (
    <div
      className="glass rounded-2xl overflow-hidden animate-slide-up group hover:border-[var(--border-strong)] transition-all duration-300 hover:shadow-lg"
      style={{ animationDelay: `${animDelay}ms` }}
    >
      {/* Rank badge */}
      <div className="flex items-start justify-between p-4 pb-0">
        <div className="flex items-center gap-2">
          <span
            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
            style={{
              background: ranked.rank === 1
                ? "linear-gradient(135deg, var(--gold), var(--accent-orange))"
                : ranked.rank === 2
                ? "linear-gradient(135deg, #c0c0c0, #a0a0a0)"
                : "linear-gradient(135deg, #cd7f32, #a05a28)",
              color: "#000",
            }}
          >
            #{ranked.rank}
          </span>
          <span className="text-xs text-[var(--text-muted)] capitalize">{p.category}</span>
        </div>
        <div className={`text-xs px-2 py-1 rounded-full font-medium ${trust.cls}`}>
          <Shield size={10} className="inline mr-1" />
          {trust.label}
        </div>
      </div>

      <div className="p-4">
        {/* Product name */}
        <h3 className="font-semibold text-sm leading-snug mb-2 group-hover:text-[var(--accent-green)] transition-colors line-clamp-2">
          {p.name}
        </h3>

        {/* Brand + Seller */}
        {(p.brand || p.seller) && (
          <p className="text-xs text-[var(--text-muted)] mb-3">
            {p.brand && <span className="font-medium text-[var(--text-secondary)]">{p.brand}</span>}
            {p.seller && p.brand && " · "}
            {p.seller && <span>{p.seller}</span>}
          </p>
        )}

        {/* Price */}
        <div className="flex items-baseline gap-2 mb-3">
          <span className="naira-price text-xl font-bold text-[var(--text-primary)]">
            {formatNaira(p.price)}
          </span>
          {p.old_price && (
            <span className="naira-price text-sm text-[var(--text-muted)] line-through">
              {formatNaira(p.old_price)}
            </span>
          )}
          {p.discount_pct && (
            <span className="text-xs font-bold text-[var(--accent-green)] bg-[var(--accent-green-dim)] px-2 py-0.5 rounded-full">
              -{p.discount_pct}%
            </span>
          )}
        </div>

        {/* Rating */}
        {p.rating && (
          <div className="flex items-center gap-1.5 mb-3">
            <div className="flex">
              {[1, 2, 3, 4, 5].map((s) => (
                <Star
                  key={s}
                  size={11}
                  className={s <= Math.round(p.rating!) ? "text-[var(--gold)]" : "text-[var(--border-strong)]"}
                  fill={s <= Math.round(p.rating!) ? "currentColor" : "none"}
                />
              ))}
            </div>
            <span className="text-xs text-[var(--text-secondary)]">{p.rating}</span>
            {p.num_reviews && (
              <span className="text-xs text-[var(--text-muted)]">({p.num_reviews.toLocaleString()})</span>
            )}
          </div>
        )}

        {/* Price Fairness Bar */}
        {p.price_fairness && (
          <div className="mb-3">
            <PriceFairnessBar fairness={p.price_fairness} />
          </div>
        )}

        {/* Trust flags */}
        {p.trust_score?.flags && p.trust_score.flags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {p.trust_score.flags.map((flag) => (
              <span key={flag} className="text-xs px-2 py-0.5 rounded-full trust-low font-medium">
                ⚠ {flag.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        )}

        {/* Composite score bar */}
        <div className="mb-3">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-[var(--text-muted)]">Match Score</span>
            <span className="text-[var(--accent-blue)] font-medium">
              {(ranked.composite_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-1 bg-[var(--surface)] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${ranked.composite_score * 100}%`,
                background: "linear-gradient(90deg, var(--accent-blue), var(--accent-purple))",
              }}
            />
          </div>
        </div>

        {/* Explanation card */}
        {explanation && (
          <div className="border-t border-[var(--border)] pt-3">
            <button
              onClick={() => setShowExplanation(!showExplanation)}
              className="flex items-center gap-1.5 text-xs text-[var(--accent-blue)] hover:text-[var(--accent-purple)] transition-colors"
            >
              <Zap size={11} />
              {showExplanation ? "Hide" : "Why recommended?"}
            </button>
            {showExplanation && (
              <div className="mt-2 space-y-1.5 animate-slide-up">
                <p className="text-xs font-medium text-[var(--text-secondary)]">{explanation.headline}</p>
                {explanation.reasons.map((r, i) => (
                  <p key={i} className="text-xs text-[var(--text-muted)] flex items-start gap-1.5">
                    <span className="text-[var(--accent-green)] mt-0.5 shrink-0">✓</span>
                    {r}
                  </p>
                ))}
                {explanation.nigerian_context && (
                  <p className="text-xs text-[var(--gold)] bg-[var(--gold-dim)] px-2 py-1.5 rounded-lg mt-2">
                    🇳🇬 {explanation.nigerian_context}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2 mt-3">
          {p.product_url && (
            <a
              href={p.product_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-[var(--naija-green)] hover:bg-[var(--accent-green)] text-white transition-all duration-200 hover:shadow-[0_0_15px_rgba(0,214,143,0.3)]"
            >
              View on Jumia
              <ExternalLink size={11} />
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ msg }: { msg: ChatMessage }) {
  const expMap = Object.fromEntries((msg.explanations || []).map((e) => [e.product_id, e]));

  return (
    <div className="animate-slide-up">
      {/* Text response */}
      <div className="glass rounded-2xl rounded-tl-sm px-4 py-3 mb-3 max-w-2xl">
        <p className="text-sm leading-relaxed text-[var(--text-primary)]">{msg.content}</p>
      </div>

      {/* Agent steps */}
      {msg.agentSteps && <AgentSteps steps={msg.agentSteps} />}

      {/* Product recommendations */}
      {msg.recommendations && msg.recommendations.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={14} className="text-[var(--accent-green)]" />
            <span className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
              Top Recommendations
            </span>
            <span className="text-xs text-[var(--text-muted)] bg-[var(--surface)] px-2 py-0.5 rounded-full">
              {msg.recommendations.length} products found
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {msg.recommendations.slice(0, 6).map((r, i) => (
              <ProductCard
                key={r.product.id}
                ranked={r}
                explanation={expMap[r.product.id]}
                animDelay={i * 80}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Page Component ────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let sid = sessionStorage.getItem("naijashop_session_id");
    if (!sid) {
      sid = `session_${Math.random().toString(36).slice(2, 11)}`;
      sessionStorage.setItem("naijashop_session_id", sid);
    }
    setSessionId(sid);
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: `u_${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const resp = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text, stream: false }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      const assistantMsg: ChatMessage = {
        id: `a_${Date.now()}`,
        role: "assistant",
        content: data.message,
        recommendations: data.recommendations || [],
        explanations: data.explanations || [],
        agentSteps: data.agent_steps || [],
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg: ChatMessage = {
        id: `e_${Date.now()}`,
        role: "assistant",
        content: "Abeg, connection issue don happen. Make sure the backend is running and try again!",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[var(--background)]">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="glass-strong border-b border-[var(--border)] px-6 py-4 shrink-0 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center text-lg font-bold"
              style={{ background: "linear-gradient(135deg, var(--naija-green), var(--accent-green))" }}
            >
              🛒
            </div>
            <div>
              <h1 className="font-bold text-base leading-none gradient-text">NaijaShop AI</h1>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                7-Agent Commerce Intelligence · Jumia · Konga
              </p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4 text-xs text-[var(--text-muted)]">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-green)]" style={{ animation: "pulse-dot 2s infinite" }} />
              DeBERTa Price Model Active
            </div>
            <div className="flex items-center gap-1.5">
              <Shield size={11} className="text-[var(--accent-blue)]" />
              Trust & Value Analysis
            </div>
          </div>
        </div>
      </header>

      {/* ── Messages ───────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 py-6">
          {isEmpty ? (
            /* Welcome Screen */
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center animate-fade-in">
              <div
                className="w-20 h-20 rounded-3xl flex items-center justify-center text-4xl mb-6 shadow-2xl"
                style={{ background: "linear-gradient(135deg, var(--naija-green), var(--accent-green))" }}
              >
                🇳🇬
              </div>
              <h2 className="text-3xl font-bold mb-3">
                <span className="gradient-text">Your Nigerian</span>
                <br />
                Shopping Intelligence
              </h2>
              <p className="text-[var(--text-secondary)] text-base max-w-md mb-8 leading-relaxed">
                Powered by 7 AI agents, live Jumia product data, and a custom-trained
                price prediction model. Find the best deals — no scam, no wahala.
              </p>

              {/* Feature badges */}
              <div className="flex flex-wrap justify-center gap-2 mb-8">
                {[
                  { icon: "🔬", label: "Price Fairness AI" },
                  { icon: "🛡️", label: "Trust Scoring" },
                  { icon: "🎯", label: "Personalized" },
                  { icon: "⚡", label: "Live Jumia Data" },
                  { icon: "💬", label: "Nigerian Context" },
                ].map(({ icon, label }) => (
                  <span
                    key={label}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full glass border border-[var(--border)]"
                  >
                    {icon} {label}
                  </span>
                ))}
              </div>

              {/* Example queries */}
              <div className="w-full max-w-2xl">
                <p className="text-xs text-[var(--text-muted)] mb-3 uppercase tracking-wider">
                  Try asking...
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {EXAMPLE_QUERIES.map((q) => (
                    <button
                      key={q}
                      onClick={() => sendMessage(q)}
                      className="text-left text-xs px-4 py-3 rounded-xl glass border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent-green)] hover:text-[var(--text-primary)] hover:bg-[var(--naija-green-dim)] transition-all duration-200"
                    >
                      "{q}"
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* Messages */
            <div className="space-y-6 max-w-7xl">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  {msg.role === "user" ? (
                    <div
                      className="max-w-md px-4 py-3 rounded-2xl rounded-tr-sm text-sm font-medium"
                      style={{ background: "linear-gradient(135deg, var(--naija-green), #005a38)", color: "white" }}
                    >
                      {msg.content}
                    </div>
                  ) : (
                    <div className="w-full">
                      <AssistantMessage msg={msg} />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="glass rounded-2xl rounded-tl-sm">
                    <TypingIndicator />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* ── Input ──────────────────────────────────────────────────────────── */}
      <div className="glass-strong border-t border-[var(--border)] p-4 shrink-0">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-3">
          <input
            ref={inputRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything — 'Find a laptop under ₦400k for ML work...'"
            disabled={isLoading}
            className="flex-1 bg-[var(--surface)] border border-[var(--border)] rounded-2xl px-5 py-3.5 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-green)] focus:ring-1 focus:ring-[var(--accent-green)] transition-all disabled:opacity-50"
          />
          <button
            type="submit"
            id="send-button"
            disabled={!input.trim() || isLoading}
            className="flex items-center justify-center w-12 h-12 rounded-2xl font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-[0_0_20px_rgba(0,214,143,0.3)]"
            style={{ background: "linear-gradient(135deg, var(--naija-green), var(--accent-green))" }}
          >
            {isLoading ? (
              <Loader2 size={18} className="text-white animate-spin" />
            ) : (
              <Send size={18} className="text-white" />
            )}
          </button>
        </form>
        <p className="text-center text-xs text-[var(--text-muted)] mt-2">
          Powered by DeBERTa price intelligence · Live Jumia data · 7 AI agents
        </p>
      </div>
    </div>
  );
}
