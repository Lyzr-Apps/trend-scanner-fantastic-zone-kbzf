'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { callAIAgent } from '@/lib/aiAgent'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import {
  HiOutlineNewspaper, HiOutlineBeaker, HiOutlineAdjustmentsHorizontal,
  HiOutlineRocketLaunch, HiOutlineArrowPath, HiOutlineCheck,
  HiOutlineExclamationTriangle, HiOutlineXMark, HiOutlineLink,
  HiOutlineChatBubbleLeftRight, HiOutlineClipboardDocument,
  HiOutlineArrowTopRightOnSquare, HiOutlineFunnel,
  HiOutlineEye, HiOutlineChartBarSquare, HiOutlineCog6Tooth,
  HiOutlineDocumentText, HiOutlineBookOpen, HiOutlineGlobeAlt,
  HiOutlineCpuChip,
  HiOutlinePencilSquare, HiOutlineCheckCircle,
  HiOutlineInformationCircle
} from 'react-icons/hi2'
import { FaXTwitter } from 'react-icons/fa6'

// --- Agent IDs ---
const MANAGER_AGENT_ID = '69995e040ab3a50ca24853ef'
const HN_AGENT_ID = '69995dde1b86f70befdb2317'
const ARXIV_AGENT_ID = '69995ddf746ef9435cac7e0e'
const CLASSIFIER_AGENT_ID = '69995d8abdf6b4ca4c1bedf7'
const TWITTER_AGENT_ID = '69995e05746ef9435cac7e1d'

// --- Interfaces ---
interface HNStory {
  title: string
  url: string
  hn_score: number
  comments_count: number
  category: string
  relevance_score: number
  summary: string
  source_type: string
}

interface ArxivPaper {
  title: string
  authors: string
  abstract_summary: string
  arxiv_link: string
  category: string
  relevance_score: number
  novelty_score: number
  applicability_score: number
}

interface ThreadDraft {
  id: string
  title: string
  classification: string
  thread_content: string
  hashtags: string
  hook: string
  requires_review: boolean
  review_reason: string
  source_url: string
  relevance_score: number
}

interface ManagerResponse {
  pipeline_status: string
  hn_results: {
    stories: HNStory[]
    total_fetched: number
    total_filtered: number
  }
  arxiv_results: {
    papers: ArxivPaper[]
    total_fetched: number
    total_filtered: number
  }
  thread_drafts: ThreadDraft[]
  total_drafts: number
  auto_approved: number
  flagged_for_review: number
  scan_timestamp: string
}

interface TwitterResponse {
  post_status: string
  tweet_url: string
  posted_content: string
  timestamp: string
  error_message: string
}

interface PublishRecord {
  draftId: string
  title: string
  status: 'pending' | 'posting' | 'success' | 'failed'
  tweetUrl: string
  timestamp: string
  errorMessage: string
}

interface AppSettings {
  relevanceThreshold: number
  categories: string[]
  sources: string[]
  autoApproveThreshold: number
  maxThreadsPerScan: number
  threadStyle: string
  blockedDomains: string
}

const DEFAULT_SETTINGS: AppSettings = {
  relevanceThreshold: 50,
  categories: ['AI/ML', 'Cybersecurity', 'Startups', 'Developer Tools', 'Research'],
  sources: ['HN Top', 'HN New', 'Ask HN', 'Show HN', 'HN Jobs', 'arXiv'],
  autoApproveThreshold: 75,
  maxThreadsPerScan: 10,
  threadStyle: 'Professional',
  blockedDomains: ''
}

// --- Deep Response Extractor ---
// The Lyzr API normalizes responses through multiple layers.
// result.response is a NormalizedAgentResponse with { status, result, message }.
// The actual schema data can be at:
//   1) result.response.result directly (ideal)
//   2) result.response.result wrapped in another { result: ... }
//   3) result.response.result as stringified JSON in a text/message field
//   4) result.response itself if normalizeResponse flattened it
//   5) result.raw_response as a string containing JSON
// This function walks every possibility and returns the first valid match.
function extractAgentData<T>(result: any, validatorKey: string): T | null {
  if (!result) return null

  // Helper: check if obj has the expected key (e.g., "pipeline_status" for Manager, "post_status" for Twitter)
  const isMatch = (obj: any): boolean => {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false
    return validatorKey in obj
  }

  // Helper: try to parse a string as JSON
  const tryParse = (str: any): any => {
    if (typeof str !== 'string') return null
    try { return JSON.parse(str) } catch { return null }
  }

  // Helper: recursively search for the schema within an object (max depth 6)
  const deepSearch = (obj: any, depth: number = 0): any => {
    if (depth > 6 || !obj) return null
    if (typeof obj === 'string') {
      const parsed = tryParse(obj)
      if (parsed && isMatch(parsed)) return parsed
      if (parsed) return deepSearch(parsed, depth + 1)
      return null
    }
    if (typeof obj !== 'object') return null
    if (isMatch(obj)) return obj

    // Check known wrapper keys
    const wrapperKeys = ['result', 'response', 'data', 'output', 'content', 'message', 'text', 'raw_response']
    for (const key of wrapperKeys) {
      if (obj[key] != null) {
        const found = deepSearch(obj[key], depth + 1)
        if (found) return found
      }
    }
    return null
  }

  // Path 1: Direct from result.response.result
  const directResult = result?.response?.result
  if (isMatch(directResult)) return directResult as T

  // Path 2: Search deeper inside result.response.result
  const deepInResult = deepSearch(directResult, 0)
  if (deepInResult) return deepInResult as T

  // Path 3: Search in result.response itself
  const fromResponse = deepSearch(result?.response, 0)
  if (fromResponse) return fromResponse as T

  // Path 4: Search in raw_response
  if (result?.raw_response) {
    const fromRaw = deepSearch(result.raw_response, 0)
    if (fromRaw) return fromRaw as T
  }

  // Path 5: Search the entire result object
  const fromTopLevel = deepSearch(result, 0)
  if (fromTopLevel) return fromTopLevel as T

  return null
}

// --- Sample Data ---
const SAMPLE_MANAGER_RESPONSE: ManagerResponse = {
  pipeline_status: 'completed',
  hn_results: {
    stories: [
      { title: 'GPT-5 Architecture Leaked: Sparse Mixture of Experts at Scale', url: 'https://news.ycombinator.com/item?id=39012345', hn_score: 847, comments_count: 312, category: 'AI/ML', relevance_score: 95, summary: 'A detailed breakdown of the rumored GPT-5 architecture reveals a massive sparse mixture-of-experts model with 1.8T parameters and novel routing mechanisms.', source_type: 'top' },
      { title: 'Show HN: Open-source alternative to Cursor IDE with local models', url: 'https://news.ycombinator.com/item?id=39012346', hn_score: 523, comments_count: 189, category: 'Developer Tools', relevance_score: 88, summary: 'A new open-source IDE that runs local LLMs for code completion, refactoring, and debugging. Supports Llama 3, Mistral, and CodeGemma.', source_type: 'show_hn' },
      { title: 'Critical vulnerability in widely-used SSH library affects millions', url: 'https://news.ycombinator.com/item?id=39012347', hn_score: 634, comments_count: 245, category: 'Cybersecurity', relevance_score: 82, summary: 'A buffer overflow vulnerability in libssh2 allows remote code execution. Patches are being rolled out but millions of servers remain exposed.', source_type: 'top' },
      { title: 'YC W25 batch includes 15 AI agent startups', url: 'https://news.ycombinator.com/item?id=39012348', hn_score: 412, comments_count: 156, category: 'Startups', relevance_score: 76, summary: 'Y Combinator Winter 2025 batch is heavily weighted toward AI agent companies, spanning customer support, code generation, and autonomous research.', source_type: 'new' },
      { title: 'Ask HN: What is the best approach to fine-tuning for RAG?', url: 'https://news.ycombinator.com/item?id=39012349', hn_score: 287, comments_count: 198, category: 'AI/ML', relevance_score: 71, summary: 'Community discussion on best practices for fine-tuning embedding models and retrieval systems for production RAG applications.', source_type: 'ask_hn' }
    ],
    total_fetched: 150,
    total_filtered: 5
  },
  arxiv_results: {
    papers: [
      { title: 'Efficient Attention Mechanisms for Long-Context Language Models', authors: 'Zhang, Wei et al.', abstract_summary: 'This paper introduces a novel linear attention mechanism that achieves near-quadratic performance while maintaining O(n) memory complexity, enabling 1M+ token context windows.', arxiv_link: 'https://arxiv.org/abs/2502.12345', category: 'AI/ML', relevance_score: 92, novelty_score: 88, applicability_score: 85 },
      { title: 'Adversarial Robustness in Multi-Agent Systems', authors: 'Chen, Li et al.', abstract_summary: 'A comprehensive framework for evaluating and improving adversarial robustness in multi-agent reinforcement learning systems, with applications to autonomous driving.', arxiv_link: 'https://arxiv.org/abs/2502.12346', category: 'Cybersecurity', relevance_score: 78, novelty_score: 82, applicability_score: 70 },
      { title: 'Self-Improving Code Generation via Iterative Refinement', authors: 'Park, Kim et al.', abstract_summary: 'A method for LLMs to iteratively improve their own code generation through self-debugging and test-driven refinement cycles, achieving SOTA on HumanEval.', arxiv_link: 'https://arxiv.org/abs/2502.12347', category: 'AI/ML', relevance_score: 85, novelty_score: 76, applicability_score: 90 }
    ],
    total_fetched: 80,
    total_filtered: 3
  },
  thread_drafts: [
    { id: 'draft-001', title: 'GPT-5 Architecture Deep Dive', classification: 'TECH DEEP DIVE', thread_content: 'The GPT-5 architecture has been revealed, and it is a game-changer.\n\nHere is what we know:\n---\n1/ Sparse Mixture of Experts with 1.8 TRILLION parameters\n\nBut only ~200B are active at inference time. This means faster responses with more knowledge.\n---\n2/ Novel routing mechanism that dynamically selects expert combinations based on query complexity.\n\nSimple queries use fewer experts = faster + cheaper.\n---\n3/ The training data reportedly includes 15T tokens with improved data quality filtering.\n\nQuality > Quantity is the new paradigm.\n---\n4/ Early benchmarks suggest 40% improvement on complex reasoning tasks vs GPT-4.\n\nMath, coding, and multi-step reasoning see the biggest gains.\n---\nThis is the future of AI. The efficiency gains alone could democratize access to frontier models.', hashtags: '#GPT5 #AI #MachineLearning #OpenAI', hook: 'The GPT-5 architecture has been revealed, and it is a game-changer.', requires_review: false, review_reason: '', source_url: 'https://news.ycombinator.com/item?id=39012345', relevance_score: 95 },
    { id: 'draft-002', title: 'Open-Source Cursor Alternative', classification: 'TECH DEEP DIVE', thread_content: 'A new open-source IDE just dropped that runs LOCAL LLMs for code completion.\n\nNo cloud, no API costs, full privacy. Here is why this matters:\n---\n1/ Supports Llama 3, Mistral, and CodeGemma out of the box.\n\nPlug in any GGUF model and start coding.\n---\n2/ Code completion, refactoring, AND debugging all powered by local inference.\n\nYour code never leaves your machine.\n---\n3/ Built on VS Code architecture so all your extensions work.\n\nZero learning curve for existing VS Code users.\n---\nThe era of local AI-powered development is here. And it is free.', hashtags: '#OpenSource #CodingTools #AI #DevTools', hook: 'A new open-source IDE just dropped that runs LOCAL LLMs for code completion.', requires_review: false, review_reason: '', source_url: 'https://news.ycombinator.com/item?id=39012346', relevance_score: 88 },
    { id: 'draft-003', title: 'Critical SSH Vulnerability Alert', classification: 'TECH DEEP DIVE', thread_content: 'CRITICAL: A major vulnerability in libssh2 affects MILLIONS of servers worldwide.\n\nHere is what you need to know and do RIGHT NOW:\n---\n1/ Buffer overflow in libssh2 allows remote code execution.\n\nAttackers can execute arbitrary code on affected servers.\n---\n2/ Any server running libssh2 < 1.11.1 is vulnerable.\n\nThis includes many cloud providers, CI/CD pipelines, and development environments.\n---\n3/ Patches are available NOW. Update immediately.\n\napt update && apt upgrade libssh2-1\n---\nDo not wait. Patch your systems today.', hashtags: '#Cybersecurity #InfoSec #Vulnerability #SSH', hook: 'CRITICAL: A major vulnerability in libssh2 affects MILLIONS of servers worldwide.', requires_review: true, review_reason: 'Security content - verify patch details and CVE reference before posting', source_url: 'https://news.ycombinator.com/item?id=39012347', relevance_score: 82 },
    { id: 'draft-004', title: 'YC W25 AI Agent Startups', classification: 'JOB POST + PREP THREAD', thread_content: 'Y Combinator W25 just revealed 15 AI agent startups in their new batch.\n\nThis is the biggest signal for where the industry is heading:\n---\n1/ Customer support agents that handle 90% of tickets autonomously.\n\nHumans only step in for edge cases.\n---\n2/ Code generation agents that can build entire features from Jira tickets.\n\nFrom spec to PR in minutes.\n---\n3/ Autonomous research agents that can read papers, run experiments, and write reports.\n\nThe future of R&D.\n---\nIf you are building in AI agents, you are in the right place at the right time.', hashtags: '#YCombinator #Startups #AIAgents #Hiring', hook: 'Y Combinator W25 just revealed 15 AI agent startups in their new batch.', requires_review: false, review_reason: '', source_url: 'https://news.ycombinator.com/item?id=39012348', relevance_score: 76 },
    { id: 'draft-005', title: 'Breakthrough: 1M Token Context Windows', classification: 'RESEARCH SUMMARY THREAD', thread_content: 'New research just solved the long-context problem for LLMs.\n\n1 MILLION token context windows with O(n) memory. Here is the breakthrough:\n---\n1/ Novel linear attention mechanism that approximates full attention.\n\nNear-quadratic performance at a fraction of the memory cost.\n---\n2/ This means you can process entire codebases, books, or document collections in a single prompt.\n\nNo more chunking or retrieval tricks.\n---\n3/ The technique is model-agnostic and can be retrofitted to existing architectures.\n\nExpect every major model to adopt this within months.\n---\nThis paper changes everything about how we think about context in AI.', hashtags: '#AIResearch #LLM #MachineLearning #NLP', hook: 'New research just solved the long-context problem for LLMs.', requires_review: false, review_reason: '', source_url: 'https://arxiv.org/abs/2502.12345', relevance_score: 92 }
  ],
  total_drafts: 5,
  auto_approved: 4,
  flagged_for_review: 1,
  scan_timestamp: '2025-02-21T14:30:00Z'
}

// --- Markdown Renderer ---
function formatInline(text: string) {
  const parts = text.split(/\*\*(.*?)\*\*/g)
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <strong key={i} className="font-semibold text-cyan-300">{part}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  )
}

function renderMarkdown(text: string) {
  if (!text) return null
  return (
    <div className="space-y-1.5">
      {text.split('\n').map((line, i) => {
        if (line.startsWith('### '))
          return <h4 key={i} className="font-semibold text-sm mt-3 mb-1 text-slate-200">{line.slice(4)}</h4>
        if (line.startsWith('## '))
          return <h3 key={i} className="font-semibold text-base mt-3 mb-1 text-slate-100">{line.slice(3)}</h3>
        if (line.startsWith('# '))
          return <h2 key={i} className="font-bold text-lg mt-4 mb-2 text-white">{line.slice(2)}</h2>
        if (line.startsWith('- ') || line.startsWith('* '))
          return <li key={i} className="ml-4 list-disc text-sm text-slate-300">{formatInline(line.slice(2))}</li>
        if (/^\d+\.\s/.test(line))
          return <li key={i} className="ml-4 list-decimal text-sm text-slate-300">{formatInline(line.replace(/^\d+\.\s/, ''))}</li>
        if (!line.trim()) return <div key={i} className="h-1" />
        return <p key={i} className="text-sm text-slate-300">{formatInline(line)}</p>
      })}
    </div>
  )
}

// --- Category Color Map ---
function getCategoryColor(category: string): string {
  const c = (category ?? '').toLowerCase()
  if (c.includes('ai') || c.includes('ml')) return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
  if (c.includes('cyber') || c.includes('security')) return 'bg-rose-500/20 text-rose-400 border-rose-500/30'
  if (c.includes('startup')) return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
  if (c.includes('developer') || c.includes('tool')) return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
  if (c.includes('research')) return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
  return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
}

function getClassificationColor(classification: string): string {
  const c = (classification ?? '').toUpperCase()
  if (c.includes('TECH DEEP DIVE')) return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
  if (c.includes('JOB')) return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
  if (c.includes('RESEARCH')) return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
  return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
}

function getSourceBadgeColor(source: string): string {
  const s = (source ?? '').toLowerCase()
  if (s === 'top') return 'bg-emerald-500/20 text-emerald-400'
  if (s === 'new') return 'bg-cyan-500/20 text-cyan-400'
  if (s.includes('ask')) return 'bg-amber-500/20 text-amber-400'
  if (s.includes('show')) return 'bg-purple-500/20 text-purple-400'
  if (s.includes('job')) return 'bg-rose-500/20 text-rose-400'
  return 'bg-slate-500/20 text-slate-400'
}

// --- Score Bar Component ---
function ScoreBar({ value, max, label, color }: { value: number; max: number; label: string; color: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300">{value}</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// --- Stat Card ---
function StatCard({ icon, label, value, accent }: { icon: React.ReactNode; label: string; value: number | string; accent: string }) {
  return (
    <Card className="bg-slate-900 border-slate-700/50">
      <CardContent className="p-4 flex items-center gap-4">
        <div className={`p-3 rounded-xl ${accent}`}>
          {icon}
        </div>
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
}

// --- Pipeline Step ---
function PipelineStep({ step, label, isActive, isDone }: { step: number; label: string; isActive: boolean; isDone: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 ${isDone ? 'bg-emerald-500 text-white' : isActive ? 'bg-cyan-500 text-white animate-pulse' : 'bg-slate-700 text-slate-400'}`}>
        {isDone ? <HiOutlineCheck className="w-4 h-4" /> : step}
      </div>
      <span className={`text-xs transition-all duration-200 ${isDone ? 'text-emerald-400' : isActive ? 'text-cyan-400' : 'text-slate-500'}`}>
        {label}
      </span>
    </div>
  )
}

// --- Thread Tweet Display ---
function ThreadTweetDisplay({ content }: { content: string }) {
  if (!content) return null
  const tweets = content.split('---').map(t => t.trim()).filter(Boolean)
  return (
    <div className="space-y-2">
      {tweets.map((tweet, i) => (
        <div key={i} className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-3 text-sm text-slate-300">
          <div className="flex items-start gap-2">
            <span className="text-xs font-mono text-slate-500 mt-0.5 shrink-0">{i + 1}/{tweets.length}</span>
            <div className="flex-1">{renderMarkdown(tweet)}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

// --- Agent Status Panel ---
function AgentStatusPanel({ activeAgentId }: { activeAgentId: string | null }) {
  const agents = [
    { id: MANAGER_AGENT_ID, name: 'Trend Intelligence Manager', desc: 'Orchestrates pipeline' },
    { id: HN_AGENT_ID, name: 'HN Data Agent', desc: 'Fetches Hacker News' },
    { id: ARXIV_AGENT_ID, name: 'arXiv Research Agent', desc: 'Fetches research papers' },
    { id: CLASSIFIER_AGENT_ID, name: 'Content Classifier & Writer', desc: 'Classifies and drafts threads' },
    { id: TWITTER_AGENT_ID, name: 'Twitter Publisher Agent', desc: 'Posts to Twitter/X' }
  ]

  return (
    <Card className="bg-slate-900 border-slate-700/50">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <HiOutlineCpuChip className="w-4 h-4 text-cyan-400" />
          Agent Pipeline
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5 pt-0">
        {agents.map(agent => (
          <div key={agent.id} className="flex items-center gap-2 py-1">
            <div className={`w-2 h-2 rounded-full transition-all duration-300 ${activeAgentId === agent.id ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'}`} />
            <span className={`text-xs transition-all duration-200 ${activeAgentId === agent.id ? 'text-cyan-400 font-medium' : 'text-slate-400'}`}>
              {agent.name}
            </span>
            <span className="text-xs text-slate-600 ml-auto">{agent.desc}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

// --- Error Boundary ---
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false, error: '' }
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950 text-white">
          <div className="text-center p-8 max-w-md">
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-slate-400 mb-4 text-sm">{this.state.error}</p>
            <button onClick={() => this.setState({ hasError: false, error: '' })} className="px-4 py-2 bg-cyan-500 text-white rounded-lg text-sm hover:bg-cyan-600 transition-colors">
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// =============================================================================
// MAIN PAGE COMPONENT
// =============================================================================
export default function Page() {
  // --- Active Tab ---
  const [activeTab, setActiveTab] = useState('dashboard')

  // --- Scan State ---
  const [scanStatus, setScanStatus] = useState<'idle' | 'scanning' | 'completed' | 'failed'>('idle')
  const [scanStep, setScanStep] = useState(0)
  const [scanData, setScanData] = useState<ManagerResponse | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null)

  // --- Sample Data Toggle ---
  const [showSample, setShowSample] = useState(false)

  // --- Content Queue State ---
  const [approvedDraftIds, setApprovedDraftIds] = useState<Set<string>>(new Set())
  const [selectedDraftIds, setSelectedDraftIds] = useState<Set<string>>(new Set())
  const [expandedDraftIds, setExpandedDraftIds] = useState<Set<string>>(new Set())
  const [classFilter, setClassFilter] = useState('All')
  const [reviewFilter, setReviewFilter] = useState('All')

  // --- Publisher State ---
  const [publishHistory, setPublishHistory] = useState<PublishRecord[]>([])
  const [publishingIds, setPublishingIds] = useState<Set<string>>(new Set())

  // --- Settings ---
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
  const [settingsSaved, setSettingsSaved] = useState(false)

  // --- Status Messages ---
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)

  // --- Debug State ---
  const [lastRawResponse, setLastRawResponse] = useState<string | null>(null)
  const [showDebug, setShowDebug] = useState(false)

  // Load settings from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('ai-trend-settings')
      if (stored) {
        const parsed = JSON.parse(stored)
        setSettings(prev => ({ ...prev, ...parsed }))
      }
    } catch {
      // ignore
    }
  }, [])

  // Derive data based on sample toggle
  const data = showSample ? SAMPLE_MANAGER_RESPONSE : scanData
  const stories = Array.isArray(data?.hn_results?.stories) ? data.hn_results.stories : []
  const papers = Array.isArray(data?.arxiv_results?.papers) ? data.arxiv_results.papers : []
  const drafts = Array.isArray(data?.thread_drafts) ? data.thread_drafts : []

  const totalScanned = (data?.hn_results?.total_fetched ?? 0) + (data?.arxiv_results?.total_fetched ?? 0)
  const highSignal = (data?.hn_results?.total_filtered ?? 0) + (data?.arxiv_results?.total_filtered ?? 0)
  const totalDrafts = data?.total_drafts ?? 0
  const flaggedForReview = data?.flagged_for_review ?? 0

  // Filtered drafts for content queue
  const filteredDrafts = drafts.filter(d => {
    if (classFilter !== 'All') {
      const upper = (d?.classification ?? '').toUpperCase()
      if (classFilter === 'Tech Deep Dive' && !upper.includes('TECH DEEP DIVE')) return false
      if (classFilter === 'Job Post' && !upper.includes('JOB')) return false
      if (classFilter === 'Research Summary' && !upper.includes('RESEARCH')) return false
    }
    if (reviewFilter === 'Auto-Approved' && d?.requires_review) return false
    if (reviewFilter === 'Needs Review' && !d?.requires_review) return false
    return true
  })

  // Approved drafts for publisher
  const approvedDrafts = drafts.filter(d => approvedDraftIds.has(d?.id ?? ''))

  // --- Scan Handler ---
  const runScan = useCallback(async () => {
    setScanStatus('scanning')
    setScanStep(1)
    setScanError(null)
    setStatusMessage({ type: 'info', text: 'Intelligence scan in progress...' })
    setActiveAgentId(MANAGER_AGENT_ID)

    // Simulate step progression
    const stepTimer1 = setTimeout(() => setScanStep(2), 3000)
    const stepTimer2 = setTimeout(() => setScanStep(3), 8000)
    const stepTimer3 = setTimeout(() => setScanStep(4), 15000)

    try {
      const message = `Run a comprehensive intelligence scan of Hacker News and arXiv for trending AI, cybersecurity, startups, and breakthrough innovation content. Fetch, filter, score, classify, and generate Twitter thread drafts for all high-signal items. Relevance threshold: ${settings.relevanceThreshold}. Max threads: ${settings.maxThreadsPerScan}. Style: ${settings.threadStyle}.`
      const result = await callAIAgent(message, MANAGER_AGENT_ID)

      clearTimeout(stepTimer1)
      clearTimeout(stepTimer2)
      clearTimeout(stepTimer3)

      // Capture raw response for debug inspection
      try {
        setLastRawResponse(JSON.stringify(result, null, 2).slice(0, 5000))
      } catch {
        setLastRawResponse('Could not serialize response')
      }

      if (result.success) {
        // Use deep extractor to find the Manager response schema regardless of nesting
        const responseData = extractAgentData<ManagerResponse>(result, 'pipeline_status')

        if (responseData && (Array.isArray(responseData.thread_drafts) || responseData.hn_results || responseData.arxiv_results)) {
          // Ensure arrays are properly typed even if the agent returned partial data
          const sanitized: ManagerResponse = {
            pipeline_status: responseData.pipeline_status ?? 'completed',
            hn_results: {
              stories: Array.isArray(responseData.hn_results?.stories) ? responseData.hn_results.stories : [],
              total_fetched: responseData.hn_results?.total_fetched ?? 0,
              total_filtered: responseData.hn_results?.total_filtered ?? 0,
            },
            arxiv_results: {
              papers: Array.isArray(responseData.arxiv_results?.papers) ? responseData.arxiv_results.papers : [],
              total_fetched: responseData.arxiv_results?.total_fetched ?? 0,
              total_filtered: responseData.arxiv_results?.total_filtered ?? 0,
            },
            thread_drafts: Array.isArray(responseData.thread_drafts) ? responseData.thread_drafts.map((d: any, idx: number) => ({
              id: d?.id ?? `draft-${idx + 1}`,
              title: d?.title ?? 'Untitled',
              classification: d?.classification ?? 'TECH DEEP DIVE',
              thread_content: d?.thread_content ?? '',
              hashtags: d?.hashtags ?? '',
              hook: d?.hook ?? '',
              requires_review: d?.requires_review === true,
              review_reason: d?.review_reason ?? '',
              source_url: d?.source_url ?? '',
              relevance_score: typeof d?.relevance_score === 'number' ? d.relevance_score : 50,
            })) : [],
            total_drafts: responseData.total_drafts ?? (Array.isArray(responseData.thread_drafts) ? responseData.thread_drafts.length : 0),
            auto_approved: responseData.auto_approved ?? 0,
            flagged_for_review: responseData.flagged_for_review ?? 0,
            scan_timestamp: responseData.scan_timestamp ?? new Date().toISOString(),
          }

          setScanData(sanitized)
          setScanStatus('completed')
          setScanStep(5)
          setStatusMessage({ type: 'success', text: `Scan complete! Found ${sanitized.total_drafts} thread drafts from ${sanitized.hn_results.stories.length} HN stories and ${sanitized.arxiv_results.papers.length} arXiv papers.` })

          // Auto-approve drafts that meet threshold
          const autoApproved = new Set<string>()
          sanitized.thread_drafts.forEach(d => {
            if (!d.requires_review && d.relevance_score >= settings.autoApproveThreshold) {
              autoApproved.add(d.id)
            }
          })
          setApprovedDraftIds(autoApproved)
        } else {
          // result.success was true but we could not find schema data - try to surface what we got
          const rawText = typeof result?.response?.message === 'string' ? result.response.message
            : typeof result?.response?.result === 'string' ? result.response.result
            : typeof result?.response?.result?.text === 'string' ? result.response.result.text
            : ''
          setScanStatus('failed')
          setScanError(`Agent returned data but no structured pipeline results were found. Raw: ${rawText.slice(0, 300)}`)
          setStatusMessage({ type: 'error', text: 'Scan returned unstructured data. Try running the scan again.' })
        }
      } else {
        setScanStatus('failed')
        setScanError(result?.error ?? result?.response?.message ?? 'Unknown error occurred')
        setStatusMessage({ type: 'error', text: `Scan failed: ${result?.error ?? result?.response?.message ?? 'Unknown error'}` })
      }
    } catch (err) {
      clearTimeout(stepTimer1)
      clearTimeout(stepTimer2)
      clearTimeout(stepTimer3)
      setScanStatus('failed')
      const msg = err instanceof Error ? err.message : 'Network error'
      setScanError(msg)
      setStatusMessage({ type: 'error', text: `Scan failed: ${msg}` })
    }
    setActiveAgentId(null)
  }, [settings.relevanceThreshold, settings.maxThreadsPerScan, settings.threadStyle, settings.autoApproveThreshold])

  // --- Publish Handler ---
  const publishThread = useCallback(async (draft: ThreadDraft) => {
    const draftId = draft?.id ?? ''
    if (!draftId) return

    setPublishingIds(prev => new Set(prev).add(draftId))
    setPublishHistory(prev => [
      ...prev.filter(p => p.draftId !== draftId),
      { draftId, title: draft?.title ?? 'Untitled', status: 'posting', tweetUrl: '', timestamp: '', errorMessage: '' }
    ])
    setActiveAgentId(TWITTER_AGENT_ID)

    try {
      const message = `Post this Twitter thread:\n\n${draft?.thread_content ?? ''}\n\nHashtags: ${draft?.hashtags ?? ''}`
      const result = await callAIAgent(message, TWITTER_AGENT_ID)

      if (result.success) {
        // Use deep extractor to find Twitter response schema
        const twitterData = extractAgentData<TwitterResponse>(result, 'post_status')

        if (twitterData) {
          const isSuccess = (twitterData.post_status ?? '').toLowerCase() === 'success'
          setPublishHistory(prev => prev.map(p =>
            p.draftId === draftId
              ? { ...p, status: isSuccess ? 'success' as const : 'failed' as const, tweetUrl: twitterData.tweet_url ?? '', timestamp: twitterData.timestamp ?? new Date().toISOString(), errorMessage: twitterData.error_message ?? '' }
              : p
          ))
          if (isSuccess) {
            setStatusMessage({ type: 'success', text: `Thread "${draft?.title ?? 'Untitled'}" posted successfully!` })
          } else {
            setStatusMessage({ type: 'error', text: `Failed to post "${draft?.title ?? 'Untitled'}": ${twitterData.error_message || 'Agent reported failure'}` })
          }
        } else {
          // Agent returned success but no structured Twitter response - check if the tool actually posted
          // The Composio TWITTER tool may return confirmation in a text message
          const msgText = result?.response?.message || result?.response?.result?.text || result?.response?.result?.message || ''
          const looksSuccessful = typeof msgText === 'string' && (msgText.toLowerCase().includes('posted') || msgText.toLowerCase().includes('tweet') || msgText.toLowerCase().includes('success'))
          setPublishHistory(prev => prev.map(p =>
            p.draftId === draftId
              ? { ...p, status: looksSuccessful ? 'success' as const : 'failed' as const, tweetUrl: '', timestamp: new Date().toISOString(), errorMessage: looksSuccessful ? '' : `Unstructured response: ${String(msgText).slice(0, 200)}` }
              : p
          ))
          if (looksSuccessful) {
            setStatusMessage({ type: 'success', text: `Thread "${draft?.title ?? 'Untitled'}" appears to have been posted. Check your Twitter account.` })
          } else {
            setStatusMessage({ type: 'error', text: `Post response unclear for "${draft?.title ?? 'Untitled'}". Check your Twitter account.` })
          }
        }
      } else {
        setPublishHistory(prev => prev.map(p =>
          p.draftId === draftId
            ? { ...p, status: 'failed' as const, errorMessage: result?.error ?? result?.response?.message ?? 'Unknown error' }
            : p
        ))
        setStatusMessage({ type: 'error', text: `Failed to post: ${result?.error ?? result?.response?.message ?? 'Unknown error'}` })
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Network error'
      setPublishHistory(prev => prev.map(p =>
        p.draftId === draftId
          ? { ...p, status: 'failed' as const, errorMessage: msg }
          : p
      ))
      setStatusMessage({ type: 'error', text: `Failed to post: ${msg}` })
    }

    setPublishingIds(prev => {
      const next = new Set(prev)
      next.delete(draftId)
      return next
    })
    setActiveAgentId(null)
  }, [])

  // --- Publish All Approved ---
  const publishAllApproved = useCallback(async () => {
    for (const draft of approvedDrafts) {
      const existing = publishHistory.find(p => p.draftId === (draft?.id ?? '') && p.status === 'success')
      if (!existing) {
        await publishThread(draft)
      }
    }
  }, [approvedDrafts, publishHistory, publishThread])

  // --- Save Settings ---
  const saveSettings = useCallback(() => {
    try {
      localStorage.setItem('ai-trend-settings', JSON.stringify(settings))
      setSettingsSaved(true)
      setStatusMessage({ type: 'success', text: 'Settings saved successfully.' })
      setTimeout(() => setSettingsSaved(false), 2000)
    } catch {
      setStatusMessage({ type: 'error', text: 'Failed to save settings.' })
    }
  }, [settings])

  // --- Toggle category in settings ---
  const toggleCategory = (cat: string) => {
    setSettings(prev => ({
      ...prev,
      categories: prev.categories.includes(cat)
        ? prev.categories.filter(c => c !== cat)
        : [...prev.categories, cat]
    }))
  }

  // --- Toggle source in settings ---
  const toggleSource = (src: string) => {
    setSettings(prev => ({
      ...prev,
      sources: prev.sources.includes(src)
        ? prev.sources.filter(s => s !== src)
        : [...prev.sources, src]
    }))
  }

  // Clear status after delay
  useEffect(() => {
    if (statusMessage) {
      const timer = setTimeout(() => setStatusMessage(null), 6000)
      return () => clearTimeout(timer)
    }
  }, [statusMessage])

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-slate-950 text-white font-sans">
        {/* Status Banner */}
        {statusMessage && (
          <div className={`fixed top-0 left-0 right-0 z-50 px-4 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-all duration-300 ${statusMessage.type === 'success' ? 'bg-emerald-600' : statusMessage.type === 'error' ? 'bg-rose-600' : 'bg-cyan-600'}`}>
            {statusMessage.type === 'success' && <HiOutlineCheckCircle className="w-4 h-4" />}
            {statusMessage.type === 'error' && <HiOutlineExclamationTriangle className="w-4 h-4" />}
            {statusMessage.type === 'info' && <HiOutlineInformationCircle className="w-4 h-4" />}
            <span>{statusMessage.text}</span>
            <button onClick={() => setStatusMessage(null)} className="ml-4 hover:opacity-70">
              <HiOutlineXMark className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="max-w-7xl mx-auto px-4 py-6">
          {/* Header */}
          <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-cyan-500/20 rounded-xl">
                <HiOutlineGlobeAlt className="w-7 h-7 text-cyan-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">AI Trend Intelligence</h1>
                <p className="text-sm text-slate-400 flex items-center gap-1.5">
                  <FaXTwitter className="w-3.5 h-3.5" />
                  Twitter/X Content Engine
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Label htmlFor="sample-toggle" className="text-xs text-slate-400 cursor-pointer">Sample Data</Label>
              <Switch
                id="sample-toggle"
                checked={showSample}
                onCheckedChange={(checked) => setShowSample(checked === true)}
              />
            </div>
          </header>

          {/* Tab Navigation */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="bg-slate-900 border border-slate-700/50 p-1 rounded-xl">
              <TabsTrigger value="dashboard" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-cyan-400 text-slate-400 gap-1.5">
                <HiOutlineChartBarSquare className="w-4 h-4" />
                Dashboard
              </TabsTrigger>
              <TabsTrigger value="queue" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-cyan-400 text-slate-400 gap-1.5">
                <HiOutlineClipboardDocument className="w-4 h-4" />
                Content Queue
              </TabsTrigger>
              <TabsTrigger value="publisher" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-cyan-400 text-slate-400 gap-1.5">
                <FaXTwitter className="w-4 h-4" />
                Publisher
              </TabsTrigger>
              <TabsTrigger value="settings" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-cyan-400 text-slate-400 gap-1.5">
                <HiOutlineCog6Tooth className="w-4 h-4" />
                Settings
              </TabsTrigger>
            </TabsList>

            {/* ================================================================ */}
            {/* DASHBOARD TAB */}
            {/* ================================================================ */}
            <TabsContent value="dashboard" className="space-y-6">
              {/* Control Panel + Pipeline Status */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Control Panel */}
                <Card className="bg-slate-900 border-slate-700/50 lg:col-span-2">
                  <CardContent className="p-6">
                    <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                      <div className="flex-1">
                        <h2 className="text-lg font-semibold text-white mb-1">Intelligence Scan</h2>
                        <p className="text-sm text-slate-400">
                          Scan Hacker News and arXiv for trending content, classify, and generate Twitter threads.
                        </p>
                      </div>
                      <Button
                        onClick={runScan}
                        disabled={scanStatus === 'scanning'}
                        className="bg-cyan-500 hover:bg-cyan-600 text-white rounded-xl px-6 py-3 text-sm font-semibold transition-all duration-200 disabled:opacity-50"
                      >
                        {scanStatus === 'scanning' ? (
                          <>
                            <HiOutlineArrowPath className="w-4 h-4 animate-spin" />
                            Scanning...
                          </>
                        ) : (
                          <>
                            <HiOutlineRocketLaunch className="w-4 h-4" />
                            Run Intelligence Scan
                          </>
                        )}
                      </Button>
                    </div>

                    {/* Pipeline Steps */}
                    {scanStatus === 'scanning' && (
                      <div className="mt-6 flex flex-wrap items-center gap-4">
                        <PipelineStep step={1} label="Fetching HN..." isActive={scanStep === 1} isDone={scanStep > 1} />
                        <div className="w-6 h-px bg-slate-700 hidden sm:block" />
                        <PipelineStep step={2} label="Fetching arXiv..." isActive={scanStep === 2} isDone={scanStep > 2} />
                        <div className="w-6 h-px bg-slate-700 hidden sm:block" />
                        <PipelineStep step={3} label="Classifying..." isActive={scanStep === 3} isDone={scanStep > 3} />
                        <div className="w-6 h-px bg-slate-700 hidden sm:block" />
                        <PipelineStep step={4} label="Generating threads..." isActive={scanStep === 4} isDone={scanStep > 4} />
                        <div className="w-6 h-px bg-slate-700 hidden sm:block" />
                        <PipelineStep step={5} label="Complete" isActive={false} isDone={scanStep >= 5} />
                      </div>
                    )}

                    {scanStatus === 'completed' && (
                      <div className="mt-4 flex items-center gap-2 text-emerald-400 text-sm">
                        <HiOutlineCheckCircle className="w-5 h-5" />
                        Scan completed at {data?.scan_timestamp ?? 'N/A'}
                      </div>
                    )}

                    {scanStatus === 'failed' && scanError && (
                      <div className="mt-4 bg-rose-500/10 border border-rose-500/20 rounded-lg p-3 text-rose-400 text-sm flex items-start gap-2">
                        <HiOutlineExclamationTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                        <div className="flex-1 break-words">
                          <p className="font-medium mb-1">Scan Failed</p>
                          <p className="text-xs text-rose-300/70">{scanError}</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Agent Status */}
                <AgentStatusPanel activeAgentId={activeAgentId} />
              </div>

              {/* Stats Overview */}
              {(data || showSample) && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatCard icon={<HiOutlineEye className="w-5 h-5 text-cyan-400" />} label="Items Scanned" value={totalScanned} accent="bg-cyan-500/10" />
                  <StatCard icon={<HiOutlineFunnel className="w-5 h-5 text-emerald-400" />} label="High-Signal Items" value={highSignal} accent="bg-emerald-500/10" />
                  <StatCard icon={<HiOutlineDocumentText className="w-5 h-5 text-purple-400" />} label="Thread Drafts" value={totalDrafts} accent="bg-purple-500/10" />
                  <StatCard icon={<HiOutlineExclamationTriangle className="w-5 h-5 text-amber-400" />} label="Flagged for Review" value={flaggedForReview} accent="bg-amber-500/10" />
                </div>
              )}

              {/* Debug Panel */}
              {lastRawResponse && (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardContent className="p-4">
                    <button
                      onClick={() => setShowDebug(!showDebug)}
                      className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1.5 transition-colors"
                    >
                      <HiOutlineInformationCircle className="w-3.5 h-3.5" />
                      {showDebug ? 'Hide' : 'Show'} raw agent response (debug)
                    </button>
                    {showDebug && (
                      <pre className="mt-3 bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-400 overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap break-all">
                        {lastRawResponse}
                      </pre>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Empty State */}
              {!data && !showSample && scanStatus === 'idle' && (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardContent className="py-16 flex flex-col items-center justify-center text-center">
                    <div className="p-4 bg-slate-800 rounded-full mb-4">
                      <HiOutlineRocketLaunch className="w-8 h-8 text-slate-500" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-300 mb-2">No scan data yet</h3>
                    <p className="text-sm text-slate-500 max-w-md">
                      Run your first intelligence scan to discover trending AI, cybersecurity, and startup content from Hacker News and arXiv. Thread drafts will be auto-generated for review.
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Loading Skeleton */}
              {scanStatus === 'scanning' && !data && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map(i => (
                      <Card key={i} className="bg-slate-900 border-slate-700/50">
                        <CardContent className="p-4">
                          <div className="animate-pulse space-y-3">
                            <div className="h-3 bg-slate-700 rounded w-2/3" />
                            <div className="h-8 bg-slate-700 rounded w-1/3" />
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                  <Card className="bg-slate-900 border-slate-700/50">
                    <CardContent className="p-6">
                      <div className="animate-pulse space-y-4">
                        <div className="h-4 bg-slate-700 rounded w-1/4" />
                        <div className="h-3 bg-slate-700 rounded w-full" />
                        <div className="h-3 bg-slate-700 rounded w-5/6" />
                        <div className="h-3 bg-slate-700 rounded w-4/6" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* HN Stories Section */}
              {stories.length > 0 && (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <HiOutlineNewspaper className="w-5 h-5 text-orange-400" />
                      Hacker News Stories
                      <Badge variant="secondary" className="ml-2 bg-slate-800 text-slate-300 border-none text-xs">
                        {stories.length} items
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-slate-400 text-sm">
                      Fetched {data?.hn_results?.total_fetched ?? 0}, filtered to {data?.hn_results?.total_filtered ?? 0} high-signal items
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <ScrollArea className="h-auto max-h-[500px]">
                      <div className="space-y-3">
                        {stories.map((story, idx) => (
                          <div key={idx} className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-4 hover:border-slate-600/50 transition-all duration-200">
                            <div className="flex flex-col sm:flex-row sm:items-start gap-3">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-start gap-2 mb-2">
                                  <a href={story?.url ?? '#'} target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-cyan-300 hover:text-cyan-200 transition-colors leading-snug flex items-start gap-1.5">
                                    {story?.title ?? 'Untitled'}
                                    <HiOutlineArrowTopRightOnSquare className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                                  </a>
                                </div>
                                <p className="text-xs text-slate-400 mb-2 line-clamp-2">{story?.summary ?? ''}</p>
                                <div className="flex flex-wrap items-center gap-2">
                                  <Badge variant="outline" className={`text-xs border ${getCategoryColor(story?.category ?? '')}`}>
                                    {story?.category ?? 'Unknown'}
                                  </Badge>
                                  <Badge variant="outline" className={`text-xs border-none ${getSourceBadgeColor(story?.source_type ?? '')}`}>
                                    {story?.source_type ?? 'unknown'}
                                  </Badge>
                                </div>
                              </div>
                              <div className="flex sm:flex-col items-center sm:items-end gap-3 sm:gap-1.5 shrink-0">
                                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                                  <HiOutlineChartBarSquare className="w-3.5 h-3.5 text-orange-400" />
                                  <span className="font-medium text-orange-300">{story?.hn_score ?? 0}</span>
                                  <span>pts</span>
                                </div>
                                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                                  <HiOutlineChatBubbleLeftRight className="w-3.5 h-3.5" />
                                  <span>{story?.comments_count ?? 0}</span>
                                </div>
                                <div className="w-16">
                                  <ScoreBar value={story?.relevance_score ?? 0} max={100} label="" color="bg-cyan-500" />
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}

              {/* arXiv Papers Section */}
              {papers.length > 0 && (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <HiOutlineBeaker className="w-5 h-5 text-purple-400" />
                      arXiv Research Papers
                      <Badge variant="secondary" className="ml-2 bg-slate-800 text-slate-300 border-none text-xs">
                        {papers.length} papers
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-slate-400 text-sm">
                      Fetched {data?.arxiv_results?.total_fetched ?? 0}, filtered to {data?.arxiv_results?.total_filtered ?? 0} high-relevance papers
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {papers.map((paper, idx) => (
                        <Card key={idx} className="bg-slate-800/50 border-slate-700/30 hover:border-slate-600/50 transition-all duration-200">
                          <CardContent className="p-4 space-y-3">
                            <a href={paper?.arxiv_link ?? '#'} target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-purple-300 hover:text-purple-200 transition-colors leading-snug flex items-start gap-1.5">
                              {paper?.title ?? 'Untitled'}
                              <HiOutlineArrowTopRightOnSquare className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                            </a>
                            <p className="text-xs text-slate-500 font-medium">{paper?.authors ?? 'Unknown authors'}</p>
                            <Badge variant="outline" className={`text-xs border ${getCategoryColor(paper?.category ?? '')}`}>
                              {paper?.category ?? 'Unknown'}
                            </Badge>
                            <p className="text-xs text-slate-400 line-clamp-3">{paper?.abstract_summary ?? ''}</p>
                            <div className="space-y-1.5">
                              <ScoreBar value={paper?.relevance_score ?? 0} max={100} label="Relevance" color="bg-cyan-500" />
                              <ScoreBar value={paper?.novelty_score ?? 0} max={100} label="Novelty" color="bg-purple-500" />
                              <ScoreBar value={paper?.applicability_score ?? 0} max={100} label="Applicability" color="bg-emerald-500" />
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* ================================================================ */}
            {/* CONTENT QUEUE TAB */}
            {/* ================================================================ */}
            <TabsContent value="queue" className="space-y-6">
              {/* Filter Bar */}
              <Card className="bg-slate-900 border-slate-700/50">
                <CardContent className="p-4">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                      <HiOutlineFunnel className="w-4 h-4" />
                      <span className="font-medium">Classification:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {['All', 'Tech Deep Dive', 'Job Post', 'Research Summary'].map(f => (
                        <Button
                          key={f}
                          variant={classFilter === f ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setClassFilter(f)}
                          className={`rounded-lg text-xs ${classFilter === f ? 'bg-cyan-500 text-white border-cyan-500 hover:bg-cyan-600' : 'bg-transparent border-slate-700 text-slate-400 hover:bg-slate-800 hover:text-slate-300'}`}
                        >
                          {f}
                        </Button>
                      ))}
                    </div>
                    <Separator orientation="vertical" className="hidden sm:block h-6 bg-slate-700" />
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                      <span className="font-medium">Review:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {['All', 'Auto-Approved', 'Needs Review'].map(f => (
                        <Button
                          key={f}
                          variant={reviewFilter === f ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setReviewFilter(f)}
                          className={`rounded-lg text-xs ${reviewFilter === f ? 'bg-cyan-500 text-white border-cyan-500 hover:bg-cyan-600' : 'bg-transparent border-slate-700 text-slate-400 hover:bg-slate-800 hover:text-slate-300'}`}
                        >
                          {f}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Bulk Actions */}
              {selectedDraftIds.size > 0 && (
                <div className="flex items-center gap-3 bg-slate-900 border border-cyan-500/30 rounded-xl p-3">
                  <span className="text-sm text-cyan-400 font-medium">{selectedDraftIds.size} selected</span>
                  <Button
                    size="sm"
                    onClick={() => {
                      setApprovedDraftIds(prev => {
                        const next = new Set(prev)
                        selectedDraftIds.forEach(id => next.add(id))
                        return next
                      })
                      setSelectedDraftIds(new Set())
                      setStatusMessage({ type: 'success', text: `${selectedDraftIds.size} drafts approved.` })
                    }}
                    className="bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg text-xs"
                  >
                    <HiOutlineCheck className="w-3.5 h-3.5" />
                    Approve Selected
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => setSelectedDraftIds(new Set())}
                    variant="outline"
                    className="border-slate-700 text-slate-400 hover:bg-slate-800 rounded-lg text-xs"
                  >
                    Clear Selection
                  </Button>
                </div>
              )}

              {/* Draft Cards */}
              {filteredDrafts.length > 0 ? (
                <ScrollArea className="h-auto">
                  <div className="space-y-4">
                    {filteredDrafts.map((draft) => {
                      const draftId = draft?.id ?? ''
                      const isApproved = approvedDraftIds.has(draftId)
                      const isSelected = selectedDraftIds.has(draftId)
                      const isExpanded = expandedDraftIds.has(draftId)

                      return (
                        <Card key={draftId} className={`bg-slate-900 border transition-all duration-200 ${isApproved ? 'border-emerald-500/30' : draft?.requires_review ? 'border-amber-500/30' : 'border-slate-700/50'}`}>
                          <CardContent className="p-5">
                            <div className="flex items-start gap-3">
                              {/* Checkbox */}
                              <div className="pt-1">
                                <Checkbox
                                  checked={isSelected}
                                  onCheckedChange={(checked) => {
                                    setSelectedDraftIds(prev => {
                                      const next = new Set(prev)
                                      if (checked) next.add(draftId)
                                      else next.delete(draftId)
                                      return next
                                    })
                                  }}
                                  className="border-slate-600 data-[state=checked]:bg-cyan-500 data-[state=checked]:border-cyan-500"
                                />
                              </div>

                              <div className="flex-1 min-w-0 space-y-3">
                                {/* Header */}
                                <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                                  <Badge variant="outline" className={`text-xs border ${getClassificationColor(draft?.classification ?? '')} w-fit`}>
                                    {draft?.classification ?? 'Unknown'}
                                  </Badge>
                                  <h3 className="text-sm font-semibold text-white">{draft?.title ?? 'Untitled'}</h3>
                                  {isApproved && (
                                    <Badge className="bg-emerald-500/20 text-emerald-400 border-none text-xs w-fit">
                                      <HiOutlineCheckCircle className="w-3 h-3 mr-1" />
                                      Approved
                                    </Badge>
                                  )}
                                  {draft?.requires_review && !isApproved && (
                                    <Badge className="bg-amber-500/20 text-amber-400 border-none text-xs w-fit">
                                      <HiOutlineExclamationTriangle className="w-3 h-3 mr-1" />
                                      Needs Review
                                    </Badge>
                                  )}
                                </div>

                                {/* Hook Preview */}
                                <p className="text-sm text-slate-300 italic">{draft?.hook ?? ''}</p>

                                {/* Review Reason */}
                                {draft?.requires_review && (draft?.review_reason ?? '') && (
                                  <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5 text-xs text-amber-300 flex items-start gap-2">
                                    <HiOutlineExclamationTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                                    <span>{draft.review_reason}</span>
                                  </div>
                                )}

                                {/* Expandable Thread Content */}
                                <div>
                                  <button
                                    onClick={() => {
                                      setExpandedDraftIds(prev => {
                                        const next = new Set(prev)
                                        if (next.has(draftId)) next.delete(draftId)
                                        else next.add(draftId)
                                        return next
                                      })
                                    }}
                                    className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
                                  >
                                    <HiOutlineEye className="w-3.5 h-3.5" />
                                    {isExpanded ? 'Hide thread' : 'View full thread'}
                                  </button>
                                  {isExpanded && (
                                    <div className="mt-3">
                                      <ThreadTweetDisplay content={draft?.thread_content ?? ''} />
                                    </div>
                                  )}
                                </div>

                                {/* Meta Row */}
                                <div className="flex flex-wrap items-center gap-3">
                                  <span className="text-xs text-slate-500 flex items-center gap-1">
                                    <HiOutlineChartBarSquare className="w-3.5 h-3.5" />
                                    Relevance: {draft?.relevance_score ?? 0}
                                  </span>
                                  {(draft?.hashtags ?? '') && (
                                    <span className="text-xs text-slate-500">{draft.hashtags}</span>
                                  )}
                                  {(draft?.source_url ?? '') && (
                                    <a href={draft.source_url} target="_blank" rel="noopener noreferrer" className="text-xs text-cyan-500 hover:text-cyan-400 flex items-center gap-1">
                                      <HiOutlineLink className="w-3 h-3" />
                                      Source
                                    </a>
                                  )}
                                </div>

                                {/* Actions */}
                                <div className="flex items-center gap-2">
                                  {!isApproved && (
                                    <Button
                                      size="sm"
                                      onClick={() => {
                                        setApprovedDraftIds(prev => new Set(prev).add(draftId))
                                        setStatusMessage({ type: 'success', text: `"${draft?.title ?? 'Untitled'}" approved.` })
                                      }}
                                      className="bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg text-xs"
                                    >
                                      <HiOutlineCheck className="w-3.5 h-3.5" />
                                      Approve
                                    </Button>
                                  )}
                                  {isApproved && (
                                    <Button
                                      size="sm"
                                      onClick={() => {
                                        setApprovedDraftIds(prev => {
                                          const next = new Set(prev)
                                          next.delete(draftId)
                                          return next
                                        })
                                      }}
                                      variant="outline"
                                      className="border-slate-700 text-slate-400 hover:bg-slate-800 rounded-lg text-xs"
                                    >
                                      <HiOutlineXMark className="w-3.5 h-3.5" />
                                      Revoke
                                    </Button>
                                  )}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </ScrollArea>
              ) : (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardContent className="py-16 flex flex-col items-center justify-center text-center">
                    <div className="p-4 bg-slate-800 rounded-full mb-4">
                      <HiOutlineClipboardDocument className="w-8 h-8 text-slate-500" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-300 mb-2">No thread drafts</h3>
                    <p className="text-sm text-slate-500 max-w-md">
                      {data && drafts.length === 0
                        ? 'The scan completed but no thread drafts were generated. Check the Dashboard debug panel for raw response details, or try running the scan again.'
                        : data
                        ? 'No drafts match your current filters. Try adjusting classification or review filters.'
                        : 'Run an intelligence scan from the Dashboard to generate thread drafts for review.'}
                    </p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* ================================================================ */}
            {/* PUBLISHER TAB */}
            {/* ================================================================ */}
            <TabsContent value="publisher" className="space-y-6">
              {/* Approved Queue */}
              <Card className="bg-slate-900 border-slate-700/50">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <FaXTwitter className="w-5 h-5 text-slate-300" />
                      Approved Queue
                      <Badge variant="secondary" className="ml-2 bg-slate-800 text-slate-300 border-none text-xs">
                        {approvedDrafts.length} threads
                      </Badge>
                    </CardTitle>
                    {approvedDrafts.length > 0 && (
                      <Button
                        onClick={publishAllApproved}
                        disabled={publishingIds.size > 0}
                        className="bg-cyan-500 hover:bg-cyan-600 text-white rounded-xl text-xs"
                      >
                        {publishingIds.size > 0 ? (
                          <><HiOutlineArrowPath className="w-3.5 h-3.5 animate-spin" /> Publishing...</>
                        ) : (
                          <><HiOutlineRocketLaunch className="w-3.5 h-3.5" /> Post All Approved</>
                        )}
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  {approvedDrafts.length > 0 ? (
                    <div className="space-y-3">
                      {approvedDrafts.map((draft) => {
                        const draftId = draft?.id ?? ''
                        const isPublishing = publishingIds.has(draftId)
                        const historyEntry = publishHistory.find(p => p.draftId === draftId)
                        const isPublished = historyEntry?.status === 'success'

                        return (
                          <div key={draftId} className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center gap-3">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <Badge variant="outline" className={`text-xs border ${getClassificationColor(draft?.classification ?? '')}`}>
                                  {draft?.classification ?? 'Unknown'}
                                </Badge>
                                <h4 className="text-sm font-semibold text-white truncate">{draft?.title ?? 'Untitled'}</h4>
                              </div>
                              <p className="text-xs text-slate-400 truncate">{draft?.hook ?? ''}</p>
                              {historyEntry?.status === 'failed' && (
                                <p className="text-xs text-rose-400 mt-1">{historyEntry.errorMessage}</p>
                              )}
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              {isPublished && (
                                <Badge className="bg-emerald-500/20 text-emerald-400 border-none text-xs">
                                  <HiOutlineCheckCircle className="w-3 h-3 mr-1" />
                                  Posted
                                </Badge>
                              )}
                              {historyEntry?.status === 'failed' && (
                                <Badge className="bg-rose-500/20 text-rose-400 border-none text-xs">
                                  <HiOutlineXMark className="w-3 h-3 mr-1" />
                                  Failed
                                </Badge>
                              )}
                              {isPublishing && (
                                <Badge className="bg-cyan-500/20 text-cyan-400 border-none text-xs">
                                  <HiOutlineArrowPath className="w-3 h-3 mr-1 animate-spin" />
                                  Posting...
                                </Badge>
                              )}
                              <Button
                                size="sm"
                                onClick={() => publishThread(draft)}
                                disabled={isPublishing}
                                className="bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-xs"
                              >
                                <FaXTwitter className="w-3.5 h-3.5" />
                                {isPublished ? 'Repost' : 'Post to Twitter'}
                              </Button>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="py-12 flex flex-col items-center justify-center text-center">
                      <div className="p-4 bg-slate-800 rounded-full mb-4">
                        <FaXTwitter className="w-8 h-8 text-slate-500" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-300 mb-2">No approved threads</h3>
                      <p className="text-sm text-slate-500 max-w-md">
                        Approve thread drafts in the Content Queue tab to add them here for publishing.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Post History */}
              {publishHistory.length > 0 && (
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <HiOutlineBookOpen className="w-5 h-5 text-slate-400" />
                      Post History
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-700/50">
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-500 uppercase">Thread</th>
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-500 uppercase">Status</th>
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-500 uppercase">Tweet URL</th>
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-500 uppercase">Timestamp</th>
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-500 uppercase">Error</th>
                          </tr>
                        </thead>
                        <tbody>
                          {publishHistory.map((record, idx) => (
                            <tr key={idx} className="border-b border-slate-800/50">
                              <td className="py-2 px-3 text-slate-300 max-w-[200px] truncate">{record.title}</td>
                              <td className="py-2 px-3">
                                <Badge className={`text-xs border-none ${record.status === 'success' ? 'bg-emerald-500/20 text-emerald-400' : record.status === 'failed' ? 'bg-rose-500/20 text-rose-400' : record.status === 'posting' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-500/20 text-slate-400'}`}>
                                  {record.status}
                                </Badge>
                              </td>
                              <td className="py-2 px-3">
                                {record.tweetUrl ? (
                                  <a href={record.tweetUrl} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 text-xs flex items-center gap-1">
                                    <HiOutlineLink className="w-3 h-3" />
                                    View Tweet
                                  </a>
                                ) : (
                                  <span className="text-slate-600 text-xs">--</span>
                                )}
                              </td>
                              <td className="py-2 px-3 text-xs text-slate-500">{record.timestamp || '--'}</td>
                              <td className="py-2 px-3 text-xs text-rose-400 max-w-[200px] truncate">{record.errorMessage || '--'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* ================================================================ */}
            {/* SETTINGS TAB */}
            {/* ================================================================ */}
            <TabsContent value="settings" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Scan Configuration */}
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-base font-semibold text-white flex items-center gap-2">
                      <HiOutlineAdjustmentsHorizontal className="w-5 h-5 text-cyan-400" />
                      Scan Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Relevance Threshold */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <Label className="text-sm text-slate-300">Relevance Threshold</Label>
                        <span className="text-sm font-mono text-cyan-400">{settings.relevanceThreshold}</span>
                      </div>
                      <Slider
                        value={[settings.relevanceThreshold]}
                        onValueChange={(val) => setSettings(prev => ({ ...prev, relevanceThreshold: val[0] ?? 50 }))}
                        max={100}
                        min={0}
                        step={5}
                        className="[&_[role=slider]]:bg-cyan-500 [&_[role=slider]]:border-cyan-600"
                      />
                      <p className="text-xs text-slate-500">Items below this score will be filtered out</p>
                    </div>

                    {/* Categories */}
                    <div className="space-y-3">
                      <Label className="text-sm text-slate-300">Categories to Scan</Label>
                      <div className="flex flex-wrap gap-3">
                        {['AI/ML', 'Cybersecurity', 'Startups', 'Developer Tools', 'Research'].map(cat => (
                          <label key={cat} className="flex items-center gap-2 cursor-pointer">
                            <Checkbox
                              checked={settings.categories.includes(cat)}
                              onCheckedChange={() => toggleCategory(cat)}
                              className="border-slate-600 data-[state=checked]:bg-cyan-500 data-[state=checked]:border-cyan-500"
                            />
                            <span className="text-sm text-slate-400">{cat}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* Sources */}
                    <div className="space-y-3">
                      <Label className="text-sm text-slate-300">Sources to Scan</Label>
                      <div className="flex flex-wrap gap-3">
                        {['HN Top', 'HN New', 'Ask HN', 'Show HN', 'HN Jobs', 'arXiv'].map(src => (
                          <label key={src} className="flex items-center gap-2 cursor-pointer">
                            <Checkbox
                              checked={settings.sources.includes(src)}
                              onCheckedChange={() => toggleSource(src)}
                              className="border-slate-600 data-[state=checked]:bg-cyan-500 data-[state=checked]:border-cyan-500"
                            />
                            <span className="text-sm text-slate-400">{src}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Content Generation Settings */}
                <Card className="bg-slate-900 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-base font-semibold text-white flex items-center gap-2">
                      <HiOutlinePencilSquare className="w-5 h-5 text-purple-400" />
                      Content Generation
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Auto-approve Threshold */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <Label className="text-sm text-slate-300">Auto-Approve Threshold</Label>
                        <span className="text-sm font-mono text-emerald-400">{settings.autoApproveThreshold}</span>
                      </div>
                      <Slider
                        value={[settings.autoApproveThreshold]}
                        onValueChange={(val) => setSettings(prev => ({ ...prev, autoApproveThreshold: val[0] ?? 75 }))}
                        max={100}
                        min={0}
                        step={5}
                        className="[&_[role=slider]]:bg-emerald-500 [&_[role=slider]]:border-emerald-600"
                      />
                      <p className="text-xs text-slate-500">Threads with relevance above this will be auto-approved</p>
                    </div>

                    {/* Max Threads */}
                    <div className="space-y-2">
                      <Label className="text-sm text-slate-300">Max Threads per Scan</Label>
                      <Input
                        type="number"
                        value={settings.maxThreadsPerScan}
                        onChange={(e) => setSettings(prev => ({ ...prev, maxThreadsPerScan: parseInt(e.target.value) || 10 }))}
                        min={1}
                        max={50}
                        className="bg-slate-800 border-slate-700 text-white"
                      />
                    </div>

                    {/* Thread Style */}
                    <div className="space-y-2">
                      <Label className="text-sm text-slate-300">Thread Style</Label>
                      <div className="flex flex-wrap gap-2">
                        {['Professional', 'Casual', 'Technical'].map(style => (
                          <Button
                            key={style}
                            variant={settings.threadStyle === style ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setSettings(prev => ({ ...prev, threadStyle: style }))}
                            className={`rounded-lg text-xs ${settings.threadStyle === style ? 'bg-purple-500 text-white border-purple-500 hover:bg-purple-600' : 'bg-transparent border-slate-700 text-slate-400 hover:bg-slate-800 hover:text-slate-300'}`}
                          >
                            {style}
                          </Button>
                        ))}
                      </div>
                    </div>

                    {/* Blocked Domains */}
                    <div className="space-y-2">
                      <Label className="text-sm text-slate-300">Blocked Domains</Label>
                      <Textarea
                        value={settings.blockedDomains}
                        onChange={(e) => setSettings(prev => ({ ...prev, blockedDomains: e.target.value }))}
                        placeholder="Enter domains to block, one per line..."
                        rows={4}
                        className="bg-slate-800 border-slate-700 text-white text-sm"
                      />
                      <p className="text-xs text-slate-500">One domain per line. Content from these domains will be excluded.</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Save Button */}
              <div className="flex justify-end">
                <Button
                  onClick={saveSettings}
                  className={`rounded-xl px-6 text-sm font-semibold transition-all duration-200 ${settingsSaved ? 'bg-emerald-500 hover:bg-emerald-600' : 'bg-cyan-500 hover:bg-cyan-600'} text-white`}
                >
                  {settingsSaved ? (
                    <><HiOutlineCheckCircle className="w-4 h-4" /> Saved!</>
                  ) : (
                    <><HiOutlineCog6Tooth className="w-4 h-4" /> Save Settings</>
                  )}
                </Button>
              </div>
            </TabsContent>
          </Tabs>

          {/* Footer Agent Pipeline Info */}
          <div className="mt-8 mb-4">
            <Card className="bg-slate-900/50 border-slate-800/50">
              <CardContent className="p-4">
                <div className="flex flex-wrap items-center justify-center gap-3 text-xs text-slate-500">
                  <span className="font-medium text-slate-400">Pipeline:</span>
                  <span className="flex items-center gap-1"><HiOutlineGlobeAlt className="w-3 h-3" /> Manager</span>
                  <span className="text-slate-700">&rarr;</span>
                  <span className="flex items-center gap-1"><HiOutlineNewspaper className="w-3 h-3" /> HN Agent</span>
                  <span className="text-slate-700">+</span>
                  <span className="flex items-center gap-1"><HiOutlineBeaker className="w-3 h-3" /> arXiv Agent</span>
                  <span className="text-slate-700">&rarr;</span>
                  <span className="flex items-center gap-1"><HiOutlineDocumentText className="w-3 h-3" /> Classifier</span>
                  <span className="text-slate-700">&rarr;</span>
                  <span className="flex items-center gap-1"><FaXTwitter className="w-3 h-3" /> Publisher</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}
