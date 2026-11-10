import { useState, useEffect } from 'react'

const API_BASE = '/api'

function App() {
  const [activeView, setActiveView] = useState('home')
  const [investigations, setInvestigations] = useState([])
  const [loading, setLoading] = useState(true)
  const [newQuestion, setNewQuestion] = useState('')
  const [newDisease, setNewDisease] = useState('')
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  // Literature view state
  const [selInvId, setSelInvId] = useState(null)
  const [litQuery, setLitQuery] = useState('')
  const [litDisease, setLitDisease] = useState('')
  const [litSources, setLitSources] = useState(['pubmed', 'europepmc'])
  const [litMaxPapers, setLitMaxPapers] = useState(20)
  const [litSearching, setLitSearching] = useState(false)
  const [papers, setPapers] = useState([])
  const [paperCount, setPaperCount] = useState(0)
  const [paperLimit, setPaperLimit] = useState(100)

  // NLP view state
  const [nlpInvId, setNlpInvId] = useState(null)
  const [entities, setEntities] = useState([])
  const [relationships, setRelationships] = useState([])
  const [evidence, setEvidence] = useState([])
  const [entityFilter, setEntityFilter] = useState('all')
  const [relFilter, setRelFilter] = useState('all')
  const [nlpProcessing, setNlpProcessing] = useState(false)
  const [nlpProcessAll, setNlpProcessAll] = useState(false)
  const [nlpErr, setNlpErr] = useState(null)
  const [paperStatuses, setPaperStatuses] = useState({})

  useEffect(() => {
    fetchInvestigations()
  }, [])

  async function fetchInvestigations() {
    try {
      const res = await fetch(`${API_BASE}/investigations`)
      if (!res.ok) throw new Error('Failed to load investigations')
      setInvestigations(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function createInvestigation(e) {
    e.preventDefault()
    setError(null)
    setSuccess(null)
    try {
      const res = await fetch(`${API_BASE}/investigations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          research_question: newQuestion,
          disease: newDisease || undefined,
          max_papers: 100,
        }),
      })
      if (!res.ok) throw new Error('Failed to create investigation')
      const created = await res.json()
      setInvestigations([created, ...investigations])
      setNewQuestion('')
      setNewDisease('')
      setSuccess('Investigation created')
    } catch (e) {
      setError(e.message)
    }
  }

  async function fetchPapers() {
    if (!selInvId) return
    try {
      const res = await fetch(`${API_BASE}/investigations/${selInvId}/literature`)
      if (!res.ok) throw new Error('Failed to load papers')
      const data = await res.json()
      setPapers(data.papers)
      setPaperCount(data.count)
      setPaperLimit(data.limit)
    } catch (e) {
      setLitErr(e.message)
    }
  }

  async function searchLiterature(e) {
    e.preventDefault()
    setLitSearching(true)
    try {
      const res = await fetch(`${API_BASE}/investigations/${selInvId}/literature/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: litQuery,
          sources: litSources,
          max_papers: litMaxPapers,
          ...(litDisease ? { start_date: litDisease } : {}),
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || err.error || 'Search failed')
      }
      const data = await res.json()
      await fetchPapers()
    } catch (e) {
      setLitErr(e.message)
    } finally {
      setLitSearching(false)
    }
  }

  async function downloadPaper(paperId) {
    try {
      const res = await fetch(`${API_BASE}/investigations/${selInvId}/literature/${paperId}/download`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Download failed')
      const data = await res.json()
      if (data.success) {
        setSuccess(`Full text downloaded: ${data.file_size} bytes`)
        await fetchPapers()
      }
    } catch (e) {
      setLitErr(e.message)
    }
  }

  async function importFile(e) {
    const file = e.target.files[0]
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/investigations/${selInvId}/literature/import`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || err.error || 'Import failed')
      }
      const data = await res.json()
      setSuccess(`Imported: ${data.title} (${data.file_size} bytes)`)
      await fetchPapers()
      e.target.value = ''
    } catch (err) {
      setLitErr(err.message)
    }
  }

  async function deletePaper(paperId) {
    if (!confirm('Delete this paper?')) return
    try {
      const res = await fetch(`${API_BASE}/investigations/${selInvId}/literature/${paperId}`, {
        method: 'DELETE',
      })
      if (!res.ok) throw new Error('Delete failed')
      await fetchPapers()
      setSuccess('Paper deleted')
    } catch (e) {
      setLitErr(e.message)
    }
  }

  async function fetchNlpData() {
    if (!nlpInvId) return
    try {
      const [entRes, relRes] = await Promise.all([
        fetch(`${API_BASE}/investigations/${nlpInvId}/nlp/entities`),
        fetch(`${API_BASE}/investigations/${nlpInvId}/nlp/relationships`),
      ])
      if (entRes.ok) setEntities(await entRes.json())
      if (relRes.ok) setRelationships(await relRes.json())
    } catch (e) {
      setNlpErr(e.message)
    }
  }

  async function fetchPaperStatuses() {
    if (!nlpInvId) return
    try {
      const res = await fetch(`${API_BASE}/investigations/${nlpInvId}/literature`)
      if (!res.ok) return
      const data = await res.json()
      const statuses = {}
      data.papers.forEach(p => {
        statuses[p.paper_id] = p.processing_status || 'pending'
      })
      setPaperStatuses(statuses)
    } catch (e) {
      console.error(e)
    }
  }

  async function processPaper(paperId) {
    setNlpProcessing(true)
    try {
      const res = await fetch(`${API_BASE}/investigations/${nlpInvId}/nlp/process/${paperId}`, {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || err.error || 'Processing failed')
      }
      const data = await res.json()
      setSuccess(`Processed: ${data.entities_extracted} entities, ${data.relationships_extracted} relationships`)
      await fetchNlpData()
      await fetchPaperStatuses()
    } catch (e) {
      setNlpErr(e.message)
    } finally {
      setNlpProcessing(false)
    }
  }

  async function processAllPapers() {
    setNlpProcessAll(true)
    try {
      const res = await fetch(`${API_BASE}/investigations/${nlpInvId}/nlp/process-all`, {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || err.error || 'Process all failed')
      }
      const data = await res.json()
      setSuccess(`Processed ${data.processed}/${data.total_papers} papers`)
      await fetchNlpData()
      await fetchPaperStatuses()
    } catch (e) {
      setNlpErr(e.message)
    } finally {
      setNlpProcessAll(false)
    }
  }

  const filteredEntities = entities.filter(e =>
    entityFilter === 'all' || e.entity_type === entityFilter
  )
  const filteredRelationships = relationships.filter(r =>
    relFilter === 'all' || r.relationship_type === relFilter
  )
  const entityTypes = [...new Set(entities.map(e => e.entity_type))]
  const relTypes = [...new Set(relationships.map(r => r.relationship_type))]

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <header className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-blue-400">Drug Discovery Literature Mining</h1>
          <p className="text-slate-400 text-sm">Local-first biomedical literature mining — M1 + M2 + M3</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveView('home')}
            className={`px-4 py-2 rounded text-sm font-medium transition ${activeView === 'home' ? 'bg-blue-600 text-white' : 'bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700'}`}
          >
            Home
          </button>
          <button
            onClick={() => setActiveView('literature')}
            className={`px-4 py-2 rounded text-sm font-medium transition ${activeView === 'literature' ? 'bg-blue-600 text-white' : 'bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700'}`}
          >
            Literature Search
          </button>
          <button
            onClick={() => setActiveView('nlp')}
            className={`px-4 py-2 rounded text-sm font-medium transition ${activeView === 'nlp' ? 'bg-blue-600 text-white' : 'bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700'}`}
          >
            NLP Extraction
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded text-red-300 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="mb-4 p-3 bg-green-900/30 border border-green-500/50 rounded text-green-300 text-sm">
          {success}
        </div>
      )}

      {activeView === 'home' ? (
        <section className="max-w-2xl">
          <h2 className="text-xl font-semibold mb-4">New Investigation</h2>
          <form onSubmit={createInvestigation} className="space-y-3">
            <div>
              <label className="block text-sm text-slate-300 mb-1">Research Question</label>
              <input
                className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={newQuestion}
                onChange={(e) => setNewQuestion(e.target.value)}
                placeholder="e.g. Identify drug-target relationships for Alzheimer's disease"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-slate-300 mb-1">Disease (optional)</label>
              <input
                className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={newDisease}
                onChange={(e) => setNewDisease(e.target.value)}
                placeholder="e.g. Alzheimer's disease"
              />
            </div>
            <button
              type="submit"
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white font-medium transition"
            >
              Create Investigation
            </button>
          </form>
        </section>
      ) : null}

      {activeView === 'literature' ? (
        <section className="max-w-4xl">
          <div className="flex flex-col md:flex-row md:items-end gap-4 mb-6">
            <div>
              <h2 className="text-xl font-semibold mb-2">Literature Search</h2>
              <p className="text-slate-400 text-sm">
                Search PubMed and Europe PMC, download open-access full text, import local PDFs/TXTs.
              </p>
            </div>
            {selInvId && (
              <button
                onClick={() => setSelInvId(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white text-sm"
              >
                Change Investigation
              </button>
            )}
          </div>

          {!selInvId ? (
            <div className="bg-slate-800/50 border border-slate-700 rounded p-6">
              <h3 className="text-lg font-medium mb-4">Select Investigation</h3>
              {loading ? (
                <p className="text-slate-400">Loading investigations...</p>
              ) : investigations.length === 0 ? (
                <p className="text-slate-400">No investigations yet. Create one on the Home tab.</p>
              ) : (
                <div className="space-y-3">
                  {investigations.map((inv) => (
                    <button
                      key={inv.investigation_id}
                      onClick={() => setSelInvId(inv.investigation_id)}
                      className="w-full text-left px-4 py-3 bg-slate-700/50 border border-slate-600 rounded hover:bg-slate-600 transition"
                    >
                      <div className="font-semibold text-blue-300">#{inv.investigation_id}</div>
                      <div className="text-sm text-slate-300">{inv.research_question}</div>
                      <div className="text-xs text-slate-500 mt-1">
                        {inv.disease ? `Disease: ${inv.disease} | ` : ''}
                        Max papers: {inv.search_parameters?.max_papers ?? 100}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="flex justify-between items-end mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-medium mb-1">
                    Investigation #{selInvId} &mdash; Literature
                  </h3>
                  <div className="text-xs text-slate-500">
                    Papers: {paperCount}/{paperLimit} ({Math.max(0, paperLimit - paperCount)} remaining)
                  </div>
                </div>
                <button
                  onClick={() => fetchPapers()}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white text-sm"
                >
                  Refresh
                </button>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded p-5 mb-6">
                <form onSubmit={searchLiterature} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-slate-300 mb-1">Search Query</label>
                      <input
                        className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        value={litQuery}
                        onChange={(e) => setLitQuery(e.target.value)}
                        placeholder="e.g. Alzheimer drug target"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-300 mb-1">Sources</label>
                      <div className="space-y-2">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={litSources.includes('pubmed')}
                            onChange={(e) => {
                              const s = e.target.checked
                                ? [...litSources, 'pubmed']
                                : litSources.filter((x) => x !== 'pubmed')
                              setLitSources(s)
                            }}
                          />
                          <span className="text-sm text-slate-300">PubMed</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={litSources.includes('europepmc')}
                            onChange={(e) => {
                              const s = e.target.checked
                                ? [...litSources, 'europepmc']
                                : litSources.filter((x) => x !== 'europepmc')
                              setLitSources(s)
                            }}
                          />
                          <span className="text-sm text-slate-300">Europe PMC</span>
                        </label>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-slate-300 mb-1">Max Papers</label>
                      <input
                        type="number"
                        className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        value={litMaxPapers}
                        onChange={(e) => setLitMaxPapers(Math.min(100, Math.max(1, parseInt(e.target.value) || 1)))}
                        min="1"
                        max="100"
                      />
                    </div>
                  </div>
                  <button
                    type="submit"
                    disabled={litSearching || !litQuery || litSources.length === 0}
                    className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 rounded text-white font-medium transition"
                  >
                    {litSearching ? 'Searching...' : 'Search Literature'}
                  </button>
                </form>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="pb-2 text-xs text-slate-400 font-semibold">#</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Title</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Source</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Status</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Date</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {papers.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="py-6 text-center text-slate-500">
                          No papers yet. Search for literature above.
                        </td>
                      </tr>
                    ) : (
                      papers.map((p) => (
                        <tr key={p.paper_id} className="border-b border-slate-800 hover:bg-slate-800/30">
                          <td className="py-3 text-xs text-slate-400">#{p.paper_id}</td>
                          <td className="py-3 text-sm text-slate-200 truncate max-w-[200px]">
                            {p.title}
                          </td>
                          <td className="py-3 text-xs">
                            <span
                              className={`px-2 py-0.5 rounded text-xs ${
                                p.source_database === 'pubmed'
                                  ? 'bg-blue-900/30 text-blue-300'
                                  : 'bg-green-900/30 text-green-300'
                              }`}
                            >
                              {p.source_database}
                            </span>
                          </td>
                          <td className="py-3 text-xs">
                            <span
                              className={`px-2 py-0.5 rounded text-xs ${
                                p.access_status === 'open_access_available'
                                  ? 'bg-green-900/30 text-green-300'
                                  : p.access_status === 'download_failed'
                                    ? 'bg-red-900/30 text-red-300'
                                    : p.access_status === 'user_imported'
                                      ? 'bg-purple-900/30 text-purple-300'
                                      : 'bg-slate-900/30 text-slate-400'
                              }`}
                            >
                              {p.access_status.replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td className="py-3 text-xs text-slate-400">
                            {p.publication_date}
                          </td>
                          <td className="py-3 text-xs">
                            {p.access_status === 'open_access_available' && !p.full_text_path ? (
                              <button
                                onClick={() => downloadPaper(p.paper_id)}
                                className="px-3 py-1 bg-green-600 hover:bg-green-500 rounded text-white text-xs"
                              >
                                Download
                              </button>
                            ) : p.full_text_path ? (
                              <span className="text-green-400 text-xs">Downloaded</span>
                            ) : (
                              <span className="text-slate-500 text-xs">N/A</span>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="mt-8 bg-slate-800/50 border border-slate-700 rounded p-5">
                <h3 className="text-lg font-medium mb-4">Import Local File (PDF/TXT)</h3>
                <form onSubmit={(e) => e.preventDefault()} className="space-y-3">
                  <div className="flex gap-3">
                    <input
                      type="file"
                      accept=".pdf,.txt"
                      onChange={importFile}
                      className="block w-full text-sm text-slate-300
                        file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                        file:text-sm file:font-semibold file:bg-blue-600 file:text-white
                        hover:file:bg-blue-500"
                    />
                    <p className="text-xs text-slate-500">
                      Max 50 MB. PDF magic bytes validated.
                    </p>
                  </div>
                </form>
              </div>
            </>
          )}
        </section>
      ) : null}

      {activeView === 'nlp' ? (
        <section className="max-w-6xl">
          <div className="flex flex-col md:flex-row md:items-end gap-4 mb-6">
            <div>
              <h2 className="text-xl font-semibold mb-2">NLP Extraction</h2>
              <p className="text-slate-400 text-sm">
                Extract biomedical entities, relationships, and evidence from processed papers.
              </p>
            </div>
            {nlpInvId && (
              <button
                onClick={() => setNlpInvId(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white text-sm"
              >
                Change Investigation
              </button>
            )}
          </div>

          {!nlpInvId ? (
            <div className="bg-slate-800/50 border border-slate-700 rounded p-6">
              <h3 className="text-lg font-medium mb-4">Select Investigation</h3>
              {loading ? (
                <p className="text-slate-400">Loading investigations...</p>
              ) : investigations.length === 0 ? (
                <p className="text-slate-400">No investigations yet. Create one on the Home tab.</p>
              ) : (
                <div className="space-y-3">
                  {investigations.map((inv) => (
                    <button
                      key={inv.investigation_id}
                      onClick={() => setNlpInvId(inv.investigation_id)}
                      className="w-full text-left px-4 py-3 bg-slate-700/50 border border-slate-600 rounded hover:bg-slate-600 transition"
                    >
                      <div className="font-semibold text-blue-300">#{inv.investigation_id}</div>
                      <div className="text-sm text-slate-300">{inv.research_question}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="flex justify-between items-end mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-medium mb-1">
                    Investigation #{nlpInvId} &mdash; NLP Extraction
                  </h3>
                  <div className="text-xs text-slate-500">
                    Entities: {entities.length} | Relationships: {relationships.length}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={fetchNlpData}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white text-sm"
                  >
                    Refresh
                  </button>
                  {paperStatuses && Object.values(paperStatuses).some(s => s !== 'completed') ? (
                    <button
                      onClick={processAllPapers}
                      disabled={nlpProcessAll}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 rounded text-white text-sm"
                    >
                      {nlpProcessAll ? 'Processing all...' : 'Process All Papers'}
                    </button>
                  ) : null}
                </div>
              </div>

              {nlpErr && (
                <div className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded text-red-300 text-sm">
                  {nlpErr}
                </div>
              )}
              {nlpProcessing || nlpProcessAll ? (
                <div className="mb-4 p-3 bg-yellow-900/30 border border-yellow-500/50 rounded text-yellow-300 text-sm">
                  {nlpProcessAll ? 'Processing all papers...' : 'Processing paper...'}
                </div>
              ) : null}

              <div className="flex gap-2 mb-4">
                <label className="text-xs text-slate-400">Filter by type:</label>
                <button
                  onClick={() => setEntityFilter('all')}
                  className={`px-3 py-1 rounded text-xs text-sm transition ${entityFilter === 'all' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300'}`}
                >
                  All
                </button>
                {entityTypes.map(t => (
                  <button
                    key={t}
                    onClick={() => setEntityFilter(t)}
                    className={`px-3 py-1 rounded text-xs text-sm transition ${entityFilter === t ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300'}`}
                  >
                    {t}
                  </button>
                ))}
              </div>

              <div className="mb-8 overflow-x-auto">
                <h4 className="text-sm font-semibold text-slate-300 mb-2">Extracted Entities ({filteredEntities.length})</h4>
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Entity</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Type</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Mentions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredEntities.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="py-6 text-center text-slate-500">
                          No entities yet. Process papers to extract entities.
                        </td>
                      </tr>
                    ) : (
                      filteredEntities.map((e) => (
                        <tr key={e.entity_id} className="border-b border-slate-800">
                          <td className="py-3 text-sm text-slate-200">{e.normalized_name}</td>
                          <td className="py-3 text-xs">
                            <span
                              className={`px-2 py-0.5 rounded text-xs ${
                                e.entity_type === 'drug' ? 'bg-blue-900/30 text-blue-300' :
                                e.entity_type === 'gene' ? 'bg-green-900/30 text-green-300' :
                                e.entity_type === 'protein' ? 'bg-purple-900/30 text-purple-300' :
                                e.entity_type === 'disease' ? 'bg-red-900/30 text-red-300' :
                                'bg-slate-900/30 text-slate-400'
                              }`}
                            >
                              {e.entity_type}
                            </span>
                          </td>
                          <td className="py-3 text-xs text-slate-400">{e.mention_count}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="mb-8 overflow-x-auto">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-sm font-semibold text-slate-300">Relationships ({filteredRelationships.length})</h4>
                  <div className="flex gap-2">
                    <label className="text-xs text-slate-400">Filter:</label>
                    <select
                      value={relFilter}
                      onChange={(e) => setRelFilter(e.target.value)}
                      className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                    >
                      <option value="all">All</option>
                      {relTypes.map(t => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Source</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Predicate</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Target</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Conf.</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Flags</th>
                      <th className="pb-2 text-xs text-slate-400 font-semibold">Context</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRelationships.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="py-6 text-center text-slate-500">
                          No relationships yet. Process papers to extract relationships.
                        </td>
                      </tr>
                    ) : (
                      filteredRelationships.map((r) => {
                        const flags = []
                        if (r.negation) flags.push('NEGATED')
                        if (r.uncertainty) flags.push('UNCERTAIN')
                        if (r.speculation) flags.push('SPECULATIVE')
                        return (
                          <tr key={r.relationship_id} className="border-b border-slate-800 hover:bg-slate-800/30">
                            <td className="py-3 text-sm text-slate-200 truncate max-w-[120px]">{r.source_name}</td>
                            <td className="py-3 text-xs">
                              <span
                                className={`px-2 py-0.5 rounded text-xs ${
                                  r.is_direct_interaction ? 'bg-blue-900/30 text-blue-300' : 'bg-slate-900/30 text-slate-400'
                                }`}
                              >
                                {r.relationship_type}
                              </span>
                            </td>
                            <td className="py-3 text-sm text-slate-200 truncate max-w-[120px]">{r.target_name}</td>
                            <td className="py-3 text-xs text-slate-400">
                              {(r.confidence_score * 100).toFixed(0)}%
                            </td>
                            <td className="py-3 text-xs">
                              {flags.length > 0 ? (
                                flags.map((f, i) => (
                                  <span key={i} className="px-1.5 py-0.5 rounded text-[10px] text-slate-200">
                                    {f}
                                  </span>
                                ))
                              ) : (
                                <span className="text-slate-500">—</span>
                              )}
                            </td>
                            <td className="py-3 text-xs text-slate-400">
                              {r.experimental_context || '—'}
                            </td>
                          </tr>
                        )
                      })
                    )}
                  </tbody>
                </table>
              </div>

              <div className="mt-8 bg-slate-800/50 border border-slate-700 rounded p-5">
                <h4 className="text-lg font-medium mb-4">Evidence Viewer</h4>
                <p className="text-sm text-slate-400 mb-4">
                  Select a relationship above to view its supporting evidence passages with full provenance.
                </p>
                {evidence.length === 0 ? (
                  <div className="text-center py-6 text-slate-500">
                    No evidence selected. Click on a relationship row to view evidence.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {evidence.map((e) => (
                      <div key={e.evidence_id} className="bg-slate-900/50 border border-slate-700 rounded p-4">
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-xs font-semibold text-blue-300">Evidence #{e.evidence_id}</span>
                          <span className="text-xs text-slate-500">Confidence: {(e.confidence_score * 100).toFixed(0)}%</span>
                        </div>
                        <p className="text-sm text-slate-200 italic mb-2">"{e.supporting_text}"</p>
                        <div className="text-xs text-slate-400 space-y-1">
                          <span>Paper: #{e.paper_id}</span>
                          <span>Section: {e.section_id || '—'} | Page: {e.page_number || '—'}</span>
                          <span>Method: {e.extraction_method}</span>
                          <span>Type: {e.evidence_type || '—'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      ) : null}

      <footer className="mt-12 pt-6 border-t border-slate-800 text-xs text-slate-500">
        Drug Discovery Literature Mining System — Local-first, no cloud LLM inference
      </footer>
    </div>
  )
}

export default App
