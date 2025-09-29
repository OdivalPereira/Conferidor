import { I18N_DEFAULT_LOCALE, loadTranslations, t } from "./i18n.js";

const React = window.React;
const ReactDOM = window.ReactDOM;

if (!React || !ReactDOM) {
  throw new Error("React e ReactDOM precisam estar disponíveis antes de carregar o app.");
}

const {useState,useEffect,useMemo,useCallback,useRef,useContext,Fragment} = React;

const DEFAULT_API_BASE = (() => {
  if (typeof window !== "undefined") {
    if (typeof window.UI_API_BASE !== "undefined") {
      const configured = String(window.UI_API_BASE);
      return configured.replace(/\/$/, "");
    }
    try {
      const params = new URLSearchParams(window.location.search);
      const param = params.get("api");
      if (param) {
        return param.replace(/\/$/, "");
      }
    } catch (err) {
      console.warn("Falha ao ler parametro ?api:", err);
    }
  }
  return "http://localhost:8000";
})();
// Ajuste `window.UI_API_BASE` ou adicione ?api=http://host:porta na URL antes de carregar o arquivo
// para definir outro endpoint.

const STATUS_COLORS = {
  "OK": {bg:"#d1fae5", badge:"#16a34a", text:"#052e16"},
  "ALERTA": {bg:"#fef3c7", badge:"#f59e0b", text:"#451a03"},
  "DIVERGENCIA": {bg:"#fee2e2", badge:"#dc2626", text:"#450a0a"},
  "SEM_FONTE": {bg:"#e5e7eb", badge:"#6b7280", text:"#111827"},
  "SEM_SUCESSOR": {bg:"#e5e7eb", badge:"#6b7280", text:"#111827"},
};

const STATUS_KEY = {
  OK: 'ok',
  ALERTA: 'alerta',
  DIVERGENCIA: 'divergencia',
  SEM_FONTE: 'sem_fonte',
  SEM_SUCESSOR: 'sem_sucessor'
};

const FILTER_DEFAULTS = Object.freeze({status:"", fonte_tipo:"", cfop:"", q:""});
const GRID_CACHE_KEY = "conferidor:grid:last-state";

function createDefaultFilters(){
  return {...FILTER_DEFAULTS};
}

function getSafeLocalStorage(){
  if(typeof window === "undefined"){ return null; }
  try{
    return window.localStorage || null;
  }catch(err){
    console.warn("LocalStorage indisponível:", err);
    return null;
  }
}

function normalizeFiltersState(input){
  const defaults = createDefaultFilters();
  if(!input || typeof input !== "object"){ return defaults; }
  const normalized = {...defaults};
  Object.keys(defaults).forEach(key=>{
    if(Object.prototype.hasOwnProperty.call(input, key)){
      const value = input[key];
      if(typeof value === "string"){ normalized[key] = value; }
      else if(value == null){ normalized[key] = ""; }
      else{ normalized[key] = String(value); }
    }
  });
  return normalized;
}

function loadCachedGridState(){
  const storage = getSafeLocalStorage();
  if(!storage){ return null; }
  try{
    const raw = storage.getItem(GRID_CACHE_KEY);
    if(!raw){ return null; }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  }catch(err){
    console.warn("Falha ao ler cache da grade:", err);
    return null;
  }
}

function saveCachedGridState(state){
  const storage = getSafeLocalStorage();
  if(!storage){ return; }
  try{
    storage.setItem(GRID_CACHE_KEY, JSON.stringify(state));
  }catch(err){
    console.warn("Falha ao salvar cache da grade:", err);
  }
}

function clearCachedGridState(){
  const storage = getSafeLocalStorage();
  if(!storage){ return; }
  try{
    storage.removeItem(GRID_CACHE_KEY);
  }catch(err){
    console.warn("Falha ao limpar cache da grade:", err);
  }
}

const REQUIRED_UPLOADS = [
  {type: "sucessor", labelKey: "upload.fields.sucessor.label", labelFallback: "Sucessor (contábil)", hintKey: "upload.fields.sucessor.hint", hintFallback: "Ex: sucessor.csv"},
  {type: "entradas", labelKey: "upload.fields.entradas.label", labelFallback: "Suprema Entradas", hintKey: "upload.fields.entradas.hint", hintFallback: "Ex: suprema_entradas.csv"},
  {type: "saidas", labelKey: "upload.fields.saidas.label", labelFallback: "Suprema Saídas", hintKey: "upload.fields.saidas.hint", hintFallback: "Ex: suprema_saidas.csv"},
  {type: "servicos", labelKey: "upload.fields.servicos.label", labelFallback: "Suprema Serviços", hintKey: "upload.fields.servicos.hint", hintFallback: "Ex: suprema_servicos.csv"},
  {type: "fornecedores", labelKey: "upload.fields.fornecedores.label", labelFallback: "Fornecedores", hintKey: "upload.fields.fornecedores.hint", hintFallback: "Ex: fornecedores.csv"},
  {type: "plano", labelKey: "upload.fields.plano.label", labelFallback: "Plano de Contas", hintKey: "upload.fields.plano.hint", hintFallback: "Ex: plano_contas.csv"}
];

const JOB_ACTIVE_STATES = new Set(["queued","running","pending","processing","in_progress","cancelling"]);

const DEFAULT_COLUMNS = [
  {id:"strategy",labelKey:"table.columns.strategy", labelFallback:"Regra",type:"text"},
  {id:"score",labelKey:"table.columns.score", labelFallback:"Score",type:"number"},
  {id:"S.data",labelKey:"table.columns.sData", labelFallback:"Data (S)",type:"date"},
  {id:"S.debito",labelKey:"table.columns.sDebito", labelFallback:"Débito",type:"text"},
  {id:"S.credito",labelKey:"table.columns.sCredito", labelFallback:"Crédito",type:"text"},
  {id:"S.doc",labelKey:"table.columns.sDoc", labelFallback:"Nº Docto (S)",type:"text"},
  {id:"S.valor",labelKey:"table.columns.sValor", labelFallback:"Valor (S)",type:"money"},
  {id:"F.doc",labelKey:"table.columns.fDoc", labelFallback:"Nº Docto (F)",type:"text"},
  {id:"F.valor",labelKey:"table.columns.fValor", labelFallback:"Valor (F)",type:"money"},
  {id:"F.cfop",labelKey:"table.columns.cfop", labelFallback:"CFOP",type:"text"},
  {id:"delta.valor",labelKey:"table.columns.deltaValor", labelFallback:"Delta Valor",type:"money"},
  {id:"delta.dias",labelKey:"table.columns.deltaDias", labelFallback:"Delta Dias",type:"number"},
  {id:"motivos",labelKey:"table.columns.motivos", labelFallback:"Motivos",type:"text"}
];

const ToastContext = React.createContext(null);

const TOAST_TONES = {
  success: {border:"border-emerald-500/40", indicator:"bg-emerald-500", title:"text-emerald-900", description:"text-emerald-700"},
  error: {border:"border-rose-500/40", indicator:"bg-rose-500", title:"text-rose-900", description:"text-rose-700"},
  warning: {border:"border-amber-500/40", indicator:"bg-amber-500", title:"text-amber-900", description:"text-amber-700"},
  info: {border:"border-slate-500/30", indicator:"bg-slate-500", title:"text-slate-900", description:"text-slate-600"}
};

function ToastViewport({toasts,removeToast}){
  if(!toasts || !toasts.length){
    return null;
  }
  return React.createElement("div",{className:"pointer-events-none fixed top-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2 px-4"},
    toasts.map(toast=>{
      if(!toast){
        return null;
      }
      const tone = TOAST_TONES[toast.type] || TOAST_TONES.info;
      return React.createElement("div",{key:toast.id,className:`pointer-events-auto rounded-xl border ${tone.border} bg-white shadow-soft`},
        React.createElement("div",{className:"flex items-start gap-3 p-4"},
          React.createElement("span",{className:`mt-1 inline-flex h-2.5 w-2.5 shrink-0 rounded-full ${tone.indicator}`},""),
          React.createElement("div",{className:"flex-1"},
            toast.title ? React.createElement("p",{className:`text-sm font-semibold ${tone.title}`},toast.title) : null,
            toast.description ? React.createElement("p",{className:`mt-1 text-sm ${tone.description}`},toast.description) : null
          ),
          React.createElement("button",{
            type:"button",
            onClick:()=>removeToast(toast.id),
            className:"ml-2 text-xs font-semibold text-slate-400 transition hover:text-slate-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          },t("common.close","Fechar"))
        )
      );
    }).filter(Boolean)
  );
}

function ToastProvider({children}){
  const [toasts,setToasts] = useState([]);
  const timeoutMap = useRef(new Map());

  const clearTimeoutById = useCallback((id)=>{
    const timeouts = timeoutMap.current;
    if(!timeouts){
      return;
    }
    const timeoutId = timeouts.get(id);
    if(timeoutId){
      clearTimeout(timeoutId);
      timeouts.delete(id);
    }
  },[]);

  const removeToast = useCallback((id)=>{
    if(!id){
      return;
    }
    setToasts(prev=>prev.filter(toast=>toast && toast.id !== id));
    clearTimeoutById(id);
  },[clearTimeoutById]);

  const addToast = useCallback((toast)=>{
    if(!toast){
      return null;
    }
    const id = toast.id || `toast_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`;
    const duration = typeof toast.duration === "number" ? toast.duration : 5000;
    const entry = {
      id,
      type: toast.type || "info",
      title: toast.title || null,
      description: toast.description || null,
      duration
    };
    setToasts(prev=>[...prev,entry]);
    if(duration !== Infinity && duration > 0){
      const timeoutId = setTimeout(()=>removeToast(id),duration);
      timeoutMap.current.set(id,timeoutId);
    }
    return id;
  },[removeToast]);

  useEffect(()=>{
    return ()=>{
      const timeouts = timeoutMap.current;
      if(timeouts){
        timeouts.forEach(timeoutId=>clearTimeout(timeoutId));
        timeouts.clear();
      }
    };
  },[]);

  const contextValue = useMemo(()=>({addToast,removeToast}),[addToast,removeToast]);

  return React.createElement(ToastContext.Provider,{value:contextValue},
    children,
    React.createElement(ToastViewport,{toasts,removeToast})
  );
}

function useToast(){
  const ctx = useContext(ToastContext);
  if(!ctx){
    throw new Error("useToast deve ser usado dentro de ToastProvider");
  }
  return ctx;
}

function resolveApiUrl(base, path){
  if(!path){
    return null;
  }
  const href = String(path);
  if(/^https?:\/\//i.test(href)){
    return href;
  }
  const safeBase = (base || "").replace(/\/$/, "");
  const safePath = href.replace(/^\//, "");
  return safeBase ? `${safeBase}/${safePath}` : `/${safePath}`;
}

function JobProgressPanel({jobId,status,apiBase,onCancel,canceling}){
  if(!jobId || !status){
    return null;
  }
  const normalized = String(status.status || "").toLowerCase();
  const progress = status.progress || {};
  const statusLabelValue = (status.status || "").replace(/_/g," ");
  const headingText = t("job.progress.title","Processamento em andamento (job {jobId})",{jobId});
  const statusText = t("job.progress.status","Status: {status}",{status: statusLabelValue});
  const downloadLogsLabel = t("job.progress.downloadLogs","Baixar logs");
  const cancelRequestedLabel = t("job.progress.cancelRequested","Cancelamento solicitado");
  const cancelingLabel = t("job.progress.canceling","Cancelando...");
  const cancelLabel = t("job.progress.cancel","Cancelar");
  const progressLabel = t("job.progress.progress","Progresso");
  const cancelNotice = t("job.progress.cancelNotice","Cancelamento em andamento — aguarde a finalização.");
  let percentValue = null;
  if(progress && Object.prototype.hasOwnProperty.call(progress,"percent")){
    const raw = progress.percent;
    const num = typeof raw === "number" ? raw : Number(raw);
    if(!Number.isNaN(num)){
      percentValue = Math.max(0, Math.min(100, num));
    }
  }
  if(percentValue === null && typeof progress.completed === "number" && typeof progress.total === "number" && progress.total){
    const computed = (progress.completed / progress.total) * 100;
    if(!Number.isNaN(computed)){
      percentValue = Math.max(0, Math.min(100, computed));
    }
  }
  const percentText = percentValue !== null ? `${percentValue.toFixed(Math.abs(percentValue) < 10 ? 1 : 0)}%` : null;
  const completed = typeof progress.completed === "number" ? progress.completed : null;
  const total = typeof progress.total === "number" ? progress.total : null;
  const steps = Array.isArray(progress.steps) && progress.steps.length ? progress.steps : (Array.isArray(status.logs) ? status.logs : []);
  const lastLog = Array.isArray(status.logs) && status.logs.length ? status.logs[status.logs.length - 1] : null;
  const currentMessage = status.message || (lastLog && (lastLog.message || lastLog.label || lastLog.name)) || null;
  const cancelRequested = Boolean(status.cancel_requested) || normalized === "cancelling";
  const disableCancel = cancelRequested || normalized === "success" || normalized === "error" || normalized === "cancelled";
  const logUrl = resolveApiUrl(apiBase, status.log_url);

  const renderStep = (step, index)=>{
    if(step==null){
      return null;
    }
    const key = typeof step === "object" && step ? step.id || step.name || step.label || step.timestamp || index : index;
    let level = null;
    let text = null;
    if(typeof step === "string"){
      text = step;
    }else if(typeof step === "object"){
      level = step.level || step.status_level || step.status;
      const parts = [];
      if(step.timestamp){ parts.push(step.timestamp); }
      const core = step.message || step.label || step.name;
      if(core){ parts.push(core); }
      if(step.status && step.status !== core){ parts.push(String(step.status)); }
      if(step.detail){ parts.push(step.detail); }
      if(parts.length === 0 && typeof step.completed === "number" && typeof step.total === "number"){
        parts.push(`${step.completed}/${step.total}`);
      }
      if(parts.length === 0 && step.message == null){
        try{
          parts.push(JSON.stringify(step));
        }catch(err){
          parts.push(String(step));
        }
      }
      text = parts.join(" — ");
    }
    if(!text){
      text = String(step);
    }
    const tone = String(level || "").toLowerCase();
    let toneClass = "text-slate-700";
    if(tone.includes("error")){
      toneClass = "text-rose-600";
    }else if(tone.includes("warn")){
      toneClass = "text-amber-600";
    }
    return React.createElement("li",{key:`step-${key}-${index}`,className:`text-sm ${toneClass}`},text);
  };

  return (
    React.createElement("div",{className:"mb-4 rounded-xl border border-indigo-200 bg-white p-4 shadow-soft"},
      React.createElement("div",{className:"flex flex-col gap-2 md:flex-row md:items-center md:justify-between"},
        React.createElement("div",{className:"space-y-1"},
          React.createElement("p",{className:"text-sm font-medium text-indigo-900"}, headingText),
          React.createElement("p",{className:"text-xs uppercase tracking-wide text-indigo-600"}, statusText)
        ),
        React.createElement("div",{className:"flex items-center gap-2"},
          logUrl && React.createElement("a",{href:logUrl,target:"_blank",rel:"noopener",className:"text-sm font-medium text-indigo-600 hover:text-indigo-500"},downloadLogsLabel),
          React.createElement("button",{
            onClick:onCancel,
            disabled:disableCancel || canceling,
            className:"rounded-lg border border-indigo-500 px-3 py-1 text-sm font-semibold text-indigo-600 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:border-slate-300 disabled:text-slate-400"
          }, cancelRequested ? cancelRequestedLabel : (canceling ? cancelingLabel : cancelLabel))
        )
      ),
      percentValue !== null && React.createElement("div",{className:"mt-4"},
        React.createElement("div",{className:"mb-1 flex justify-between text-xs text-slate-500"},
          React.createElement("span",null,progressLabel),
          React.createElement("span",null,
            percentText,
            completed !== null && total !== null ? ` · ${completed}/${total}` : ""
          )
        ),
        React.createElement("div",{className:"h-2 w-full overflow-hidden rounded-full bg-slate-200"},
          React.createElement("div",{className:"h-full bg-indigo-500 transition-all",style:{width:`${percentValue}%`}}
          )
        )
      ),
      currentMessage && React.createElement("p",{className:"mt-4 text-sm text-slate-600"}, currentMessage),
      steps.length ? React.createElement("div",{className:"mt-4 max-h-48 overflow-y-auto rounded-lg bg-slate-50 p-3"},
        React.createElement("ol",{className:"space-y-1"}, steps.slice(-12).map(renderStep).filter(Boolean))
      ) : null,
      status.log_size ? React.createElement("p",{className:"mt-3 text-xs text-slate-400"}, t("job.progress.logSize","Log: {size} KB",{size:(status.log_size/1024).toFixed(1)})) : null,
      cancelRequested ? React.createElement("p",{className:"mt-3 text-xs text-amber-600"},cancelNotice) : null
    )
  );
}

function Badge({status}){
  const m = STATUS_COLORS[status] || STATUS_COLORS["SEM_FONTE"];
  return React.createElement("span",{className:"badge",style:{background:m.badge,color:m.text}},status||"—");
}

function useApi(base=DEFAULT_API_BASE){
  const baseUrl = base ? String(base).replace(/\/$/, "") : "";

  const isPlainObject = (value)=> Object.prototype.toString.call(value) === "[object Object]";

  const mergeHeaders = (...inputs)=>{
    const headers = new Headers();
    inputs.filter(Boolean).forEach(input=>{
      if(input instanceof Headers){
        input.forEach((value,key)=>{ headers.set(key,value); });
        return;
      }
      if(Array.isArray(input)){
        input.forEach(entry=>{
          if(Array.isArray(entry) && entry.length >= 2){
            headers.set(entry[0], entry[1]);
          }
        });
        return;
      }
      if(typeof input === "object"){
        Object.entries(input).forEach(([key,value])=>{
          if(value != null){
            headers.set(key, value);
          }
        });
      }
    });
    return headers;
  };

  const buildUrl = (path, params)=>{
    const qs = params && Object.keys(params).length ? new URLSearchParams(Object.entries(params).filter(([,value])=> value !== undefined && value !== null)).toString() : "";
    return baseUrl + path + (qs ? ("?" + qs) : "");
  };

  const request = async (path, options={})=>{
    const {method="GET", params, body, headers: extraHeaders, init: rawInit, ...rest} = options || {};
    const url = buildUrl(path, params || {});
    const init = {...(rawInit || {}), ...rest, method};
    const headers = mergeHeaders(rawInit?.headers, extraHeaders);

    if(typeof body !== "undefined"){
      if(body instanceof FormData){
        init.body = body;
        headers.delete("Content-Type");
      }else if(isPlainObject(body)){
        init.body = JSON.stringify(body);
        if(!headers.has("Content-Type")){
          headers.set("Content-Type","application/json");
        }
      }else{
        init.body = body;
      }
    }

    if([...headers.keys()].length){
      init.headers = headers;
    }

    const response = await fetch(url, init);
    const text = await response.text();
    if(!response.ok){
      throw new Error(text || response.statusText || "Request failed");
    }
    if(!text){
      return {};
    }
    try{
      return JSON.parse(text);
    }catch(err){
      return {raw:text};
    }
  };

  const get = (path, options)=> request(path,{...(options||{}), method:"GET"});
  const post = (path, options)=> request(path,{...(options||{}), method:"POST"});
  const del = (path, options)=> request(path,{...(options||{}), method:"DELETE"});

  return {get,post,del,request};
}

function UploadModal({open,onClose,onConfirm,submitting,error}){
  const [files,setFiles] = useState({});
  const [missing,setMissing] = useState([]);
  const [messages,setMessages] = useState([]);

  useEffect(()=>{
    if(open){
      setFiles({});
      setMissing([]);
      setMessages([]);
    }
  },[open]);

  if(!open){
    return null;
  }

  const handleFileChange = (type,fileList)=>{
    const file = fileList && fileList.length ? fileList[0] : null;
    setFiles(prev=>({...prev,[type]:file}));
    setMissing(prev=>prev.filter(item=>item!==type));
  };

  const handleClose = ()=>{
    if(submitting){
      return;
    }
    onClose();
  };

  const handleSubmit = async (event)=>{
    if(event && typeof event.preventDefault === "function"){ event.preventDefault(); }
    const requiredMissing = REQUIRED_UPLOADS.filter(item=>!files[item.type]);
    if(requiredMissing.length){
      setMissing(requiredMissing.map(item=>item.type));
      setMessages(requiredMissing.map(item=>{
        const labelText = t(item.labelKey,item.labelFallback);
        return t("upload.validation.selectFile","Selecione o arquivo de {label}.",{label: labelText});
      }));
      return;
    }
    setMessages([]);
    const formData = new FormData();
    REQUIRED_UPLOADS.forEach(item=>{
      const file = files[item.type];
      if(file){
        formData.append("files",file,file.name);
      }
    });
    try{
      await onConfirm(formData);
    }catch(err){
      setMessages([String(err)]);
    }
  };

  const combinedMessages = error ? [...messages,String(error)] : messages;

  return (
    React.createElement("div",{className:"fixed inset-0 z-50 bg-slate-900/60 flex items-center justify-center px-4"},
      React.createElement("div",{className:"relative w-full max-w-2xl rounded-2xl bg-white shadow-2xl"},
        React.createElement("button",{className:"absolute right-4 top-4 text-slate-400 hover:text-slate-600",onClick:handleClose},
          "✕"
        ),
        React.createElement("form",{onSubmit:handleSubmit,className:"p-6 md:p-8"},
          React.createElement("div",{className:"mb-6 space-y-2"},
            React.createElement("h3",{className:"text-xl font-semibold text-slate-900"},t("upload.title","Enviar CSVs")),
            React.createElement("p",{className:"text-sm text-slate-600"},t("upload.subtitle","Selecione os arquivos CSV necessários para iniciar o processamento.")),
          ),
          React.createElement("div",{className:"grid gap-4"},
            REQUIRED_UPLOADS.map(item=>{
              const hasError = missing.includes(item.type);
              const borderClass = hasError ? "border-rose-400 focus:border-rose-500 focus:ring-rose-200" : "border-slate-200 focus:border-slate-400 focus:ring-slate-200";
              const labelText = t(item.labelKey,item.labelFallback);
              const hintText = t(item.hintKey,item.hintFallback);
              return React.createElement("label",{key:item.type,className:"flex flex-col gap-2 rounded-xl border bg-slate-50/60 p-4"},
                React.createElement("div",{className:"flex flex-col"},
                  React.createElement("span",{className:"text-sm font-semibold text-slate-700"}, labelText),
                  React.createElement("span",{className:"text-xs text-slate-500"}, hintText)
                ),
                React.createElement("input",{
                  type:"file",
                  accept:".csv",
                  className:`block w-full rounded-lg border bg-white px-3 py-2 text-sm shadow-sm focus:outline-none ${borderClass}`,
                  onChange:e=>handleFileChange(item.type,e.target.files)
                })
              );
            })
          ),
          combinedMessages.length ? React.createElement("div",{className:"mt-6 space-y-1 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"},
            combinedMessages.map((msg,idx)=>React.createElement("div",{key:idx},msg))
          ) : null,
          React.createElement("div",{className:"mt-8 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end"},
            React.createElement("button",{
              type:"button",
              onClick:handleClose,
              disabled:submitting,
              className:"rounded-lg bg-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-300 disabled:opacity-60"
            },t("common.cancel","Cancelar")),
            React.createElement("button",{
              type:"submit",
              disabled:submitting,
              className:"rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
            }, submitting?t("upload.actions.submitting","Enviando..."):t("upload.actions.submit","Enviar e processar"))
          )
        )
      )
    )
  );
}

function Toolbar({stats,filters,onStatusFilter,onFilterChange,onReload,onExportJson,onExportX,onExportP,onClearFilters,onClearData,clearing,loading,disabled}){
  const statusValue = (filters?.status || "");
  const statusCards = [
    {key:"TOTAL",labelKey:"toolbar.cards.total",label:"Total",valueKey:"TOTAL",statusValue:"",valueClass:"text-xl font-bold"},
    {key:"OK",labelKey:"toolbar.cards.ok",label:"OK",valueKey:"OK",statusValue:"OK",valueClass:"text-xl font-bold text-emerald-700"},
    {key:"ALERTA",labelKey:"toolbar.cards.alert",label:"Alertas",valueKey:"ALERTA",statusValue:"ALERTA",valueClass:"text-xl font-bold text-amber-700"},
    {key:"DIVERGENCIA",labelKey:"toolbar.cards.divergence",label:"Divergências",valueKey:"DIVERGENCIA",statusValue:"DIVERGENCIA",valueClass:"text-xl font-bold text-rose-700"},
    {key:"SEM_FONTE",labelKey:"toolbar.cards.noSource",label:"Sem Fonte",valueKey:"SEM_FONTE",statusValue:"SEM_FONTE",valueClass:"text-xl font-bold text-slate-700"},
    {key:"SEM_SUCESSOR",labelKey:"toolbar.cards.noSuccessor",label:"Sem Sucessor",valueKey:"SEM_SUCESSOR",statusValue:"SEM_SUCESSOR",valueClass:"text-xl font-bold text-slate-700"}
  ];
  return (
    React.createElement("div",{className:"flex flex-col gap-3 md:flex-row md:items-end md:justify-between mb-4"},
      React.createElement("div",{className:"grid grid-cols-2 md:grid-cols-6 gap-3"},
        statusCards.map(card=>{
          const isActive = statusValue === (card.statusValue || "");
          const baseClass = "p-3 rounded-xl bg-white shadow-soft border-2 transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60";
          const stateClass = isActive ? " border-indigo-500 bg-indigo-50" : " border-transparent hover:border-slate-200";
          return React.createElement("button",{
            key: card.key,
            type:"button",
            className: baseClass + stateClass,
            onClick: ()=>{ if(onStatusFilter){ onStatusFilter(card.statusValue || ""); } },
            disabled,
            role:"button",
            "aria-pressed": isActive
          },[
            React.createElement("div",{key:"label",className:"text-xs text-slate-500"}, t(card.labelKey,card.label)),
            React.createElement("div",{key:"value",className: card.valueClass}, stats?.[card.valueKey] ?? "—")
          ]);
        })
      ),
      React.createElement("div",{className:"flex flex-col gap-2 md:w-[520px]"},
        React.createElement("div",{className:"flex gap-2"},
          React.createElement("select",{className:"px-2 py-2 rounded-lg bg-white border w-40 disabled:opacity-50",value:filters.status,disabled,onChange:e=>onStatusFilter && onStatusFilter(e.target.value)},
            React.createElement("option",{value:""},t("toolbar.filters.statusAll","Status — todos")),
            ["OK","ALERTA","DIVERGENCIA","SEM_FONTE","SEM_SUCESSOR"].map(s=>React.createElement("option",{key:s,value:s},s))
          ),
          React.createElement("select",{className:"px-2 py-2 rounded-lg bg-white border w-40 disabled:opacity-50",value:filters.fonte_tipo,disabled,onChange:e=>onFilterChange && onFilterChange("fonte_tipo", e.target.value)},
            React.createElement("option",{value:""},t("toolbar.filters.sourceAll","Fonte — todas")),
            ["ENTRADA","SAIDA","SERVICO"].map(s=>React.createElement("option",{key:s,value:s},s))
          ),
          React.createElement("input",{className:"px-3 py-2 rounded-lg bg-white border flex-1 disabled:opacity-50",placeholder:t("toolbar.filters.cfop","CFOP"),value:filters.cfop,disabled,onChange:e=>onFilterChange && onFilterChange("cfop", e.target.value)})
        ),
        React.createElement("div",{className:"flex gap-2"},
          React.createElement("input",{className:"px-3 py-2 rounded-lg bg-white border flex-1 disabled:opacity-50",placeholder:t("toolbar.filters.search","Buscar (histórico, doc, tags)..."),value:filters.q,disabled,onChange:e=>onFilterChange && onFilterChange("q", e.target.value),onKeyDown:e=>{if(e.key==="Escape"){ e.preventDefault(); e.stopPropagation(); if(onFilterChange){ onFilterChange("q",""); } e.target.blur(); }}}),
          React.createElement("button",{onClick:onReload,disabled:loading||disabled,className:"px-3 py-2 rounded-lg bg-slate-900 text-white disabled:opacity-50 disabled:cursor-not-allowed"}, loading?t("common.loading","Carregando..."):t("toolbar.actions.refresh","Atualizar"))
        ),
        React.createElement("div",{className:"flex gap-2"},
          React.createElement("button",{onClick:onClearFilters,disabled:disabled,className:"px-3 py-2 rounded-lg bg-slate-200 text-slate-700 hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"}, t("toolbar.actions.clearFilters","Limpar filtros")),
          React.createElement("button",{onClick:onExportJson,disabled:disabled,className:"px-3 py-2 rounded-lg bg-amber-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"}, t("toolbar.actions.exportJson","Exportar JSON")),
          React.createElement("button",{onClick:onExportX,disabled:disabled,className:"px-3 py-2 rounded-lg bg-emerald-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"}, t("toolbar.actions.exportExcel","Exportar Excel")),
          React.createElement("button",{onClick:onExportP,disabled:disabled,className:"px-3 py-2 rounded-lg bg-indigo-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"}, t("toolbar.actions.exportPdf","Exportar PDF")),
          React.createElement("button",{onClick:onClearData,disabled:disabled||clearing,className:"px-3 py-2 rounded-lg bg-rose-600 text-white hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed"}, clearing?t("toolbar.actions.clearing","Limpando..."):t("toolbar.actions.clearData","Limpar dados")),
          React.createElement("a",{href:"/files/",target:"_blank",className:"px-3 py-2 rounded-lg bg-slate-200"}, t("toolbar.actions.files","Arquivos"))
        )
      )
    )
  );
}

function InspectorPanel({row,onClose,onMark,onReset}){
  if(!row){
    return null;
  }
  const info = [
    {label: t("inspector.fields.currentStatus","Status atual"), value: row.status || row["match.status"]},
    {label: t("inspector.fields.strategy","Estrategia"), value: row["match.strategy"]},
    {label: t("inspector.fields.score","Score"), value: row["match.score"]},
    {label: t("inspector.fields.sucessor","Sucessor"), value: row["S.doc"]},
    {label: t("inspector.fields.source","Fonte"), value: row["F.doc"]},
    {label: t("inspector.fields.deltaValue","Delta Valor"), value: row["delta.valor"]},
    {label: t("inspector.fields.deltaDays","Delta Dias"), value: row["delta.dias"]},
    {label: t("inspector.fields.reasons","Motivos"), value: row.motivos}
  ];
  return React.createElement("div",{className:"mt-6 bg-white shadow-soft rounded-xl border border-slate-200"},
    React.createElement("div",{className:"flex items-start justify-between px-4 py-3 border-b border-slate-200"},
      React.createElement("h3",{className:"text-lg font-semibold"},t("inspector.title","Detalhes da selecao")),
      React.createElement("button",{className:"text-sm text-slate-500 hover:text-slate-700",onClick:onClose},t("common.close","Fechar"))
    ),
    React.createElement("div",{className:"px-4 py-4 grid gap-3 md:grid-cols-2"},
      info.map((item,idx)=>React.createElement("div",{key:idx},
        React.createElement("div",{className:"text-xs uppercase tracking-wide text-slate-500"}, item.label),
        React.createElement("div",{className:"text-sm font-medium text-slate-800 break-words"}, item.value || "-")
      ))
    ),
    React.createElement("div",{className:"px-4 pb-4 flex flex-wrap gap-2"},
      React.createElement("button",{className:"px-3 py-1 rounded bg-emerald-100 text-emerald-800 text-sm",onClick:()=>onMark(row,"OK")},t("inspector.actions.markOk","Marcar OK")),
      React.createElement("button",{className:"px-3 py-1 rounded bg-rose-100 text-rose-800 text-sm",onClick:()=>onMark(row,"DIVERGENCIA")},t("inspector.actions.markDivergence","Marcar Divergencia")),
      React.createElement("button",{className:"px-3 py-1 rounded bg-slate-200 text-slate-700 text-sm",onClick:()=>onReset(row)},t("inspector.actions.reset","Reverter"))
    )
  );
}

function Table({columns,items,resolveRowId,selectedRowId,sortBy,setSortBy,sortDir,setSortDir,onInspect,onManualStatus,onResetStatus,onSortChange,loading,emptyMessage,emptyHint}){
  const setSort = useCallback((id)=>{
    let nextSortBy = id;
    let nextSortDir = "desc";
    if(sortBy === id){
      nextSortDir = sortDir === "asc" ? "desc" : "asc";
    }
    if(typeof onSortChange === "function"){
      onSortChange(nextSortBy, nextSortDir);
      return;
    }
    if(typeof setSortBy === "function"){
      setSortBy(nextSortBy);
    }
    if(typeof setSortDir === "function"){
      setSortDir(nextSortDir);
    }
  },[onSortChange, setSortBy, setSortDir, sortBy, sortDir]);
  const tableItems = Array.isArray(items) ? items : [];
  const getColumnLabel = useCallback((column)=>{
    if(!column){
      return "";
    }
    const fallback = column.label || column.labelFallback || column.id;
    if(column.labelKey){
      return t(column.labelKey, fallback);
    }
    return fallback;
  },[]);
  const statusHeader = t("table.headers.status","Status");
  const actionsHeader = t("table.headers.actions","Acoes");
  const markOkLabel = t("table.actions.markOk","Marcar OK");
  const markDivergenceLabel = t("table.actions.markDivergence","Marcar Divergencia");
  const resetLabel = t("table.actions.reset","Reverter");
  const loadingTitle = t("table.loading.title","Carregando dados...");
  const loadingDescription = t("table.loading.description","Estamos preparando a grade para exibição.");
  const hasItems = tableItems.length > 0;
  const showEmptyState = !loading && !hasItems;
  const emptyLabel = emptyMessage || t("table.empty.title","Nenhum registro encontrado.");
  const emptyDetails = emptyHint || t("table.empty.description","Ajuste os filtros ou processe novos dados.");
  return (
    React.createElement("div",{className:"relative min-h-[240px]", "aria-busy": loading ? "true" : "false"},
      React.createElement("div",{className:"overflow-auto rounded-xl shadow-soft border bg-white h-full"},
        React.createElement("table",{className:"min-w-[1200px] w-full table-sticky"},
          React.createElement("thead",null,
            React.createElement("tr",null,
              React.createElement("th",{className:"text-left px-3 py-2 w-28"},statusHeader),
              React.createElement("th",{className:"text-left px-3 py-2 w-44"},actionsHeader),
              columns.map(col=>React.createElement("th",{key:col.id,className:"text-left px-3 py-2 cursor-pointer",onClick:()=>setSort(col.id)}, getColumnLabel(col)))
            )
          ),
          React.createElement("tbody",null,
          tableItems.map((r,idx)=>{
            const s = (r.status||"").toUpperCase();
            const auto = (r.score ?? 0) >= 70;
            const cls = s==="OK"?"row-ok":(s==="ALERTA"?"row-alerta":(s==="DIVERGENCIA"?"row-div":"row-neutro"));
            const rowKey = (resolveRowId && resolveRowId(r)) ?? idx;
            const isSelected = selectedRowId && rowKey === selectedRowId;
            return React.createElement("tr",{key:rowKey,className:cls + (isSelected?" ring-2 ring-indigo-400":"") + " hover:bg-slate-100",style: auto?{outline:"2px solid #2563eb",outlineOffset:"-2px"}:{} ,onClick:()=>onInspect && onInspect(r)},
              React.createElement("td",{className:"px-3 py-2 align-top"}, React.createElement(Badge,{status:s})),
              React.createElement("td",{className:"px-3 py-2 align-top"},
                React.createElement("div",{className:"flex flex-col gap-1"},
                  React.createElement("button",{className:"px-2 py-1 rounded bg-emerald-100 text-emerald-800 text-xs",onClick:(e)=>{e.stopPropagation(); onManualStatus && onManualStatus(r,"OK")}},markOkLabel),
                  React.createElement("button",{className:"px-2 py-1 rounded bg-rose-100 text-rose-800 text-xs",onClick:(e)=>{e.stopPropagation(); onManualStatus && onManualStatus(r,"DIVERGENCIA")}},markDivergenceLabel),
                  React.createElement("button",{className:"px-2 py-1 rounded bg-slate-200 text-slate-700 text-xs",disabled:!r._manual,onClick:(e)=>{e.stopPropagation(); onResetStatus && onResetStatus(r)}},resetLabel)
                )
              ),
              columns.map(col=>{
                const val = r[col.id];
                const isMoney = (col.type==="money");
                const isNum = (col.type==="number");
                const txt = (val==null)?"":(isMoney?formatBRL(val):(isNum?String(val):String(val)));
                return React.createElement("td",{key:col.id,className:"px-3 py-2 align-top "+(isMoney||isNum?"cell-mono":"")},
                  txt
                );
              })
            );
          })
        )
      )
      ),
      loading ? React.createElement("div",{className:"absolute inset-0 flex flex-col items-center justify-center gap-2 bg-white/80 backdrop-blur-sm text-indigo-700",role:"status","aria-live":"polite"},
        React.createElement("span",{className:"spinner","aria-hidden":"true"}),
        React.createElement("p",{className:"text-sm font-semibold"},loadingTitle),
        React.createElement("p",{className:"text-xs"},loadingDescription)
      ) : null,
      showEmptyState ? React.createElement("div",{className:"pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 bg-white/70 text-slate-600"},
        React.createElement("p",{className:"text-sm font-semibold"}, emptyLabel),
        React.createElement("p",{className:"text-xs text-slate-500"}, emptyDetails)
      ) : null
    )
  );
}

function Paginator({offset,setOffset,limit,setLimit,total,go}){
  const canPrev = offset>0;
  const canNext = offset+limit < total;
  const page = Math.floor(offset/limit)+1;
  const pages = Math.max(1, Math.ceil(total/limit));
  const summaryText = t("paginator.summary","Mostrando {start}-{end} de {total}",{
    start: Math.min(total, offset+1),
    end: Math.min(total, offset+limit),
    total
  });
  const pageText = t("paginator.page","Pagina {page}/{pages}",{page,pages});
  const previousLabel = t("paginator.previous","Anterior");
  const nextLabel = t("paginator.next","Próximo");
  return (
    React.createElement("div",{className:"flex items-center justify-between mt-3"},
      React.createElement("div",null, summaryText),
      React.createElement("div",{className:"flex items-center gap-2"},
        React.createElement("button",{disabled:!canPrev,onClick:()=>{ if(!canPrev) return; const nextOffset = Math.max(0, offset-limit); if(typeof go === "function"){ go(nextOffset); } },className:"px-3 py-1 rounded bg-slate-200 disabled:opacity-50", "aria-label": previousLabel},"<"),
        React.createElement("span",null, pageText),
        React.createElement("button",{disabled:!canNext,onClick:()=>{ if(!canNext) return; const nextOffset = offset+limit; if(typeof go === "function"){ go(nextOffset); } },className:"px-3 py-1 rounded bg-slate-200 disabled:opacity-50", "aria-label": nextLabel},">"),
        React.createElement("select",{className:"px-2 py-1 rounded bg-white border",value:limit,onChange:e=>{const nextLimit = parseInt(e.target.value,10); setOffset(0); setLimit(nextLimit);}},
          [50,100,200,500,1000].map(n=>React.createElement("option",{key:n,value:n}, n))
        )
      )
    )
  );
}

function formatBRL(x){
  const n = Number(x);
  if(Number.isNaN(n)) return x==null?"":String(x);
  return n.toLocaleString("pt-BR",{style:"currency",currency:"BRL"});
}

function computeToolbarStats(meta){
  const stats = meta?.stats;
  if(!stats){
    return null;
  }
  const uppercase = Object.entries(STATUS_KEY).reduce((acc,[upper,lower])=>{
    const value = stats[lower];
    acc[upper] = typeof value === "number" ? value : Number(value) || 0;
    return acc;
  },{});
  const total = Object.values(uppercase).reduce((sum,val)=> sum + (typeof val === "number" ? val : Number(val) || 0), 0);
  return {...stats, ...uppercase, TOTAL: total};
}

function App(){
  const api = useApi(); // ajuste `window.UI_API_BASE` ou use ?api= para apontar para outro host/porta
  const {addToast} = useToast();
  const defaultErrorTitle = t("toast.defaults.errorTitle","Erro");
  const defaultErrorDescription = t("toast.defaults.errorDescription","Ocorreu um erro inesperado.");
  const defaultWarningTitle = t("toast.defaults.warningTitle","Aviso");
  const defaultWarningDescription = t("toast.defaults.warningDescription","Verifique as informações exibidas.");
  const metaErrorTitle = t("app.notifications.metaError","Erro ao carregar metadados");
  const gridErrorTitle = t("app.notifications.gridError","Erro ao carregar dados");
  const gridErrorMessage = t("app.notifications.gridErrorMessage","Não foi possível carregar os dados da grade. Tente novamente.");
  const confirmClearMessage = t("app.confirm.clearData","Tem certeza que deseja limpar os dados processados? Esta ação não pode ser desfeita.");
  const clearSuccessTitle = t("app.notifications.clearSuccessTitle","Dados limpos");
  const clearSuccessMessage = t("app.notifications.clearSuccessMessage","Os dados processados foram removidos.");
  const clearErrorToastTitle = t("app.notifications.clearErrorToastTitle","Erro ao limpar dados");
  const clearErrorFallback = t("app.notifications.clearErrorFallback","Não foi possível limpar os dados processados.");
  const processingFailedTitle = t("app.notifications.processingFailedTitle","Processamento falhou");
  const processingFailedMessage = t("app.notifications.processingFailedMessage","O processamento foi interrompido.");
  const processingCancelledTitle = t("app.notifications.processingCancelledTitle","Processamento cancelado");
  const processingCancelledMessage = t("app.notifications.processingCancelledMessage","O processamento foi interrompido antes da conclusão.");
  const cancelProcessErrorTitle = t("app.notifications.cancelProcessError","Não foi possível cancelar o processamento");
  const exportJsonErrorTitle = t("app.notifications.exportJsonError","Erro ao exportar JSON");
  const exportXlsxErrorTitle = t("app.notifications.exportXlsxError","Erro ao exportar XLSX");
  const exportPdfErrorTitle = t("app.notifications.exportPdfError","Erro ao exportar PDF");
  const processStartErrorTitle = t("app.notifications.processStartError","Erro ao iniciar processamento");
  const uploadErrorTitle = t("app.notifications.uploadError","Erro no envio de arquivos");
  const manualSaveErrorTitle = t("app.notifications.manualSaveError","Falha ao salvar ajuste manual");
  const manualRemoveErrorTitle = t("app.notifications.manualRemoveError","Falha ao remover ajuste manual");
  const emptyFilteredMessage = t("app.empty.filtered","Nenhum registro encontrado para os filtros selecionados.");
  const emptyDefaultMessage = t("app.empty.default","Nenhum dado disponível no momento.");
  const emptyFilteredHint = t("app.empty.filteredHint","Ajuste os filtros ou limpe os campos para visualizar outros resultados.");
  const emptyDefaultHint = t("app.empty.defaultHint","Envie novos arquivos para gerar a grade de conferência.");
  const appTitle = t("app.header.title","App de Conferencia - UI");
  const subtitleTemplate = t("app.header.subtitle","Carrega {file} do servidor local e exibe a grade com filtros e exports.");
  const subtitleFile = t("app.header.subtitleFile","ui_grid.jsonl");
  const processButtonProcessing = t("app.actions.processing","Processando...");
  const processButtonLabel = t("app.actions.process","Processar");
  const showErrorToast = useCallback((title, error)=>{
    const description = error && error.message ? error.message : (typeof error === "string" ? error : null);
    addToast({
      type:"error",
      title: title || defaultErrorTitle,
      description: description || defaultErrorDescription
    });
  },[addToast, defaultErrorDescription, defaultErrorTitle]);
  const showWarningToast = useCallback((title, description)=>{
    addToast({
      type:"warning",
      title: title || defaultWarningTitle,
      description: description || defaultWarningDescription
    });
  },[addToast, defaultWarningDescription, defaultWarningTitle]);
  const cachedGridStateRef = useRef(loadCachedGridState());
  const cachedGridState = cachedGridStateRef.current;
  const [schema,setSchema] = useState(null);
  const [meta,setMeta] = useState(null);
  const [items,setItems] = useState(()=> Array.isArray(cachedGridState?.items) ? cachedGridState.items : []);
  const [total,setTotal] = useState(()=>{
    if(typeof cachedGridState?.total === "number"){ return cachedGridState.total; }
    const numeric = Number(cachedGridState?.total);
    return Number.isFinite(numeric) ? numeric : 0;
  });
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState(null);
  const [filters,setFilters] = useState(()=> normalizeFiltersState(cachedGridState?.filters));
  const [sortBy,setSortBy] = useState(()=>{
    const candidate = typeof cachedGridState?.sortBy === "string" ? cachedGridState.sortBy : null;
    return candidate && candidate.length ? candidate : "score";
  });
  const [sortDir,setSortDir] = useState(()=> cachedGridState?.sortDir === "asc" ? "asc" : "desc");
  const [limit,setLimit] = useState(()=>{
    const numeric = Number(cachedGridState?.limit);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : 100;
  });
  const [offset,setOffset] = useState(()=>{
    const numeric = Number(cachedGridState?.offset);
    return Number.isFinite(numeric) && numeric >= 0 ? numeric : 0;
  });
  const [selectedRow,setSelectedRow] = useState(null);
  const [showUploadModal,setShowUploadModal] = useState(false);
  const [uploading,setUploading] = useState(false);
  const [uploadError,setUploadError] = useState(null);
  const [currentJobId,setCurrentJobId] = useState(null);
  const [jobStatus,setJobStatus] = useState(null);
  const [jobCanceling,setJobCanceling] = useState(false);
  const [clearingData,setClearingData] = useState(false);
  const normalizedJobStatus = String(jobStatus?.status || "").toLowerCase();
  const jobRunning = Boolean(currentJobId) && JOB_ACTIVE_STATES.has(normalizedJobStatus);
  const skipNextAutoLoad = useRef(false);
  const offsetResetGuard = useRef(false);

  const persistGridState = useCallback((state)=>{
    const base = state && typeof state === "object" ? {...state} : {};
    base.items = Array.isArray(base.items) ? base.items : [];
    base.total = typeof base.total === "number" ? base.total : Number(base.total) || 0;
    base.limit = Number(base.limit);
    if(!Number.isFinite(base.limit) || base.limit <= 0){ base.limit = 100; }
    base.offset = Number(base.offset);
    if(!Number.isFinite(base.offset) || base.offset < 0){ base.offset = 0; }
    base.sortBy = typeof base.sortBy === "string" && base.sortBy ? base.sortBy : "score";
    base.sortDir = base.sortDir === "asc" ? "asc" : "desc";
    base.filters = normalizeFiltersState(base.filters);
    base.timestamp = typeof base.timestamp === "number" ? base.timestamp : Date.now();
    saveCachedGridState(base);
    cachedGridStateRef.current = base;
  },[cachedGridStateRef]);

  const clearGridCacheRef = useCallback(()=>{
    clearCachedGridState();
    cachedGridStateRef.current = null;
  },[cachedGridStateRef]);

  const resolveRowId = useCallback((entry)=> entry?.id ?? [entry?.sucessor_idx, entry?.fonte_tipo, entry?.fonte_idx].filter(Boolean).join('-'), []);

  const handleInspect = useCallback((row)=>{
    if(!row){
      setSelectedRow(null);
      return;
    }
    setSelectedRow(row);
  },[]);

  useEffect(()=>{
    if(!selectedRow){
      return;
    }
    const rowKey = resolveRowId(selectedRow);
    const latest = items.find(item=> resolveRowId(item) === rowKey);
    if(!latest){
      setSelectedRow(null);
    }else if(latest !== selectedRow){
      setSelectedRow(latest);
    }
  },[items, resolveRowId, selectedRow]);

  const selectedRowId = selectedRow ? resolveRowId(selectedRow) : null;

  const tableColumns = useMemo(() => schema?.columns?.length ? schema.columns : DEFAULT_COLUMNS, [schema]);

  const loadMeta = useCallback(async ()=>{
    try{
      const s = await api.get("/api/schema");
      setSchema(s);
      const m = await api.get("/api/meta");
      setMeta(m);
    }catch(e){
      console.error("Falha ao carregar metadados", e);
      showErrorToast(metaErrorTitle, e);
    }
  },[api, metaErrorTitle, showErrorToast]);

  const loadGrid = useCallback(async (nextOffset, nextFilters)=>{
    const hasOverride = typeof nextOffset === "number" && !Number.isNaN(nextOffset);
    const targetOffset = hasOverride ? Math.max(0, nextOffset) : offset;
    if(hasOverride && nextOffset !== offset){
      skipNextAutoLoad.current = true;
      setOffset(targetOffset);
    }
    const appliedFilters = nextFilters || filters;
    setLoading(true); setError(null);
    try{
      const params = {
        limit,
        offset: targetOffset,
        status: appliedFilters.status || undefined,
        fonte_tipo: appliedFilters.fonte_tipo || undefined,
        cfop: appliedFilters.cfop || undefined,
        q: appliedFilters.q || undefined,
        sort_by: sortBy || undefined,
        sort_dir: sortDir || undefined
      };
      const res = await api.get("/api/grid",{params});
      const nextItems = Array.isArray(res.items) ? res.items : [];
      const nextTotal = typeof res.total_filtered === "number" ? res.total_filtered : Number(res.total_filtered) || 0;
      setItems(nextItems);
      setTotal(nextTotal);
      persistGridState({
        items: nextItems,
        total: nextTotal,
        filters: appliedFilters,
        sortBy,
        sortDir,
        limit,
        offset: targetOffset
      });
    }catch(e){
      console.error("Falha ao carregar grade", e);
      setError(gridErrorMessage);
      setItems([]);
      setTotal(0);
      showErrorToast(gridErrorTitle, e);
    }
    finally{ setLoading(false); }
  },[api, filters, sortBy, sortDir, limit, offset, skipNextAutoLoad, gridErrorMessage, gridErrorTitle, showErrorToast, persistGridState]);

  useEffect(()=>{ loadMeta(); },[loadMeta]);
  useEffect(()=>{
    if(skipNextAutoLoad.current){
      skipNextAutoLoad.current = false;
      return;
    }
    loadGrid();
  },[loadGrid]);

  useEffect(()=>{
    if(!offsetResetGuard.current){
      offsetResetGuard.current = true;
      return;
    }
    setOffset(prev=> prev === 0 ? prev : 0);
  },[filters, sortBy, sortDir, offsetResetGuard]);

  const handleSortChange = useCallback((nextSortBy, nextSortDir)=>{
    setSortBy(nextSortBy);
    setSortDir(nextSortDir);
    setOffset(prev=> prev === 0 ? prev : 0);
  },[]);

  const handleStatusFilter = useCallback((nextStatus)=>{
    const normalized = nextStatus ? String(nextStatus).toUpperCase() : "";
    setOffset(0);
    setFilters(prev=>{
      const currentStatus = prev?.status || "";
      if(currentStatus === normalized){
        return prev;
      }
      return {...prev, status: normalized};
    });
  },[]);

  const handleFilterChange = useCallback((field, value)=>{
    setOffset(0);
    setFilters(prev=>{
      if(!prev || prev[field] === value){
        return prev;
      }
      return {...prev, [field]: value};
    });
  },[]);

  const handleClearFilters = useCallback(()=>{
    const cleared = createDefaultFilters();
    setOffset(0);
    setFilters(cleared);
    loadGrid(0, cleared);
  },[loadGrid]);

  const handleClearData = useCallback(async ()=>{
    if(clearingData){
      return;
    }
    const confirmed = window.confirm(confirmClearMessage);
    if(!confirmed){
      return;
    }
    setClearingData(true);
    setError(null);
    try{
      await api.del("/api/data");
      setMeta(null);
      setItems([]);
      setTotal(0);
      setSelectedRow(null);
      clearGridCacheRef();
      addToast({
        type:"success",
        title:clearSuccessTitle,
        description:clearSuccessMessage
      });
    }catch(err){
      console.error("Falha ao limpar dados", err);
      const message = err && err.message ? err.message : String(err);
      addToast({
        type:"error",
        title:clearErrorToastTitle,
        description: message || clearErrorFallback
      });
    }finally{
      setClearingData(false);
    }
  },[addToast, api, clearingData, clearErrorFallback, clearErrorToastTitle, clearSuccessMessage, clearSuccessTitle, confirmClearMessage, clearGridCacheRef]);

  const exportJ = async ()=>{
    try{
      const out = "reconc_grid.json";
      const res = await api.post("/api/export/json",{body:{grid:"reconc_grid.csv",out}});
      const link = res.download || res.absolute_out;
      if(link){ window.open(link,"_blank"); }
    }catch(e){
      console.error("Falha no export JSON", e);
      showErrorToast(exportJsonErrorTitle, e);
    }
  };
  const exportX = async ()=>{
    try{
      const out = "relatorio_conferencia.xlsx";
      const res = await api.post("/api/export/xlsx",{body:{grid:"reconc_grid.csv",sem_fonte:"reconc_sem_fonte.csv",sem_sucessor:"reconc_sem_sucessor.csv",out}});
      if(res.download){ window.open(res.download,"_blank"); }
    }catch(e){
      console.error("Falha no export XLSX", e);
      showErrorToast(exportXlsxErrorTitle, e);
    }
  };
  const exportP = async ()=>{
    try{
      const out = "relatorio_conferencia.pdf";
      const res = await api.post("/api/export/pdf",{body:{grid:"reconc_grid.csv",out,cliente:"Cliente",periodo:"Periodo"}});
      const link = res.download || res.download_html;
      if(link){ window.open(link,"_blank"); }
    }catch(e){
      console.error("Falha no export PDF", e);
      showErrorToast(exportPdfErrorTitle, e);
    }
  };

  const handleCancelJob = useCallback(async ()=>{
    if(!currentJobId || jobCanceling){
      return;
    }
    setJobCanceling(true);
    try{
      const res = await api.del(`/api/process/${currentJobId}`);
      setJobStatus(prev=>{
        if(prev && typeof prev === "object"){
          return {...prev, ...res};
        }
        return res;
      });
    }catch(err){
      console.error("Falha ao solicitar cancelamento do job", err);
      const message = err && err.message ? err.message : String(err);
      showErrorToast(cancelProcessErrorTitle, message);
    }finally{
      setJobCanceling(false);
    }
  },[api,cancelProcessErrorTitle,currentJobId,jobCanceling,showErrorToast]);

  const openUploadModal = ()=>{
    setUploadError(null);
    setShowUploadModal(true);
  };

  const closeUploadModal = ()=>{
    setShowUploadModal(false);
    setUploadError(null);
  };

  const handleUploadConfirm = async (formData)=>{
    setUploading(true);
    setUploadError(null);
    try{
      const uploadRes = await api.post("/api/uploads",{body:formData});
      const jobId = uploadRes?.job_id;
      if(jobId){
        let initialStatus = uploadRes?.status || null;
        try{
          const processRes = await api.post("/api/process",{body:{job_id: jobId}});
          initialStatus = processRes?.status || initialStatus || "running";
        }catch(processErr){
          const message = processErr && processErr.message ? processErr.message : String(processErr);
          setUploadError(message);
          console.error("Falha ao iniciar processamento", processErr);
          showErrorToast(processStartErrorTitle, message);
          return;
        }
        setCurrentJobId(jobId);
        setJobCanceling(false);
        setJobStatus({job_id: jobId, status: initialStatus});
        if(String(initialStatus||"").toLowerCase() === "success"){
          await loadMeta();
          await loadGrid(0);
          setCurrentJobId(null);
          setJobStatus(null);
        }
      }else{
        await loadMeta();
        await loadGrid(0);
      }
      setShowUploadModal(false);
    }catch(err){
      const message = err && err.message ? err.message : String(err);
      setUploadError(message);
      console.error("Falha no envio de arquivos", err);
      showErrorToast(uploadErrorTitle, message);
    }finally{
      setUploading(false);
    }
  };

  const updateStats = (prevStatus, nextStatus)=>{
    if(!prevStatus || !nextStatus || prevStatus===nextStatus) return;
    setMeta(current=>{
      if(!current || !current.stats) return current;
      const stats = {...current.stats};
      const prevKey = STATUS_KEY[prevStatus];
      const nextKey = STATUS_KEY[nextStatus];
      if(prevKey && stats[prevKey]!=null){ stats[prevKey] = Math.max(0, stats[prevKey]-1); }
      if(nextKey){ stats[nextKey] = (stats[nextKey]||0)+1; }
      return {...current, stats};
    });
  };

  const handleManualStatus = async (row, nextStatus)=>{
    if(!row) return;
    const targetStatus = (nextStatus || "").toUpperCase();
    if(!targetStatus) return;
    const rowKey = resolveRowId(row);
    if(!rowKey) return;
    const current = items.find(item=> resolveRowId(item) === rowKey) || row;
    const previousStatus = (current?.status || current?.["match.status"] || "").toUpperCase();
    const originalStatus = (current?.original_status || previousStatus || "").toUpperCase();
    try{
      await api.post("/api/manual-status",{
        body:{
          row_id: rowKey,
          status: targetStatus,
          original_status: originalStatus
        }
      });
    }catch(e){
      console.error("Falha ao salvar override manual", e);
      showErrorToast(manualSaveErrorTitle, e);
      return;
    }
    let updatedRow = null;
    let nextItems = null;
    setItems(prev=>{
      const mapped = prev.map(item=>{
        const key = resolveRowId(item);
        if(key !== rowKey) return item;
        const motivos = new Set((item.motivos || "").split(";").filter(Boolean));
        motivos.add("ajuste_manual");
        const updated = {
          ...item,
          status: targetStatus,
          "match.status": targetStatus,
          motivos: Array.from(motivos).join(";"),
          _manual: true,
          original_status: item.original_status || originalStatus
        };
        updatedRow = updated;
        return updated;
      });
      nextItems = mapped;
      return mapped;
    });
    if(updatedRow){ setSelectedRow(updatedRow); }
    if(nextItems){
      persistGridState({
        items: nextItems,
        total,
        filters,
        sortBy,
        sortDir,
        limit,
        offset
      });
    }
    if(previousStatus){ updateStats(previousStatus, targetStatus); }
  };

  const handleResetStatus = async (row)=>{
    if(!row) return;
    const rowKey = resolveRowId(row);
    if(!rowKey) return;
    const current = items.find(item=> resolveRowId(item) === rowKey) || row;
    const previousStatus = (current?.status || current?.["match.status"] || "").toUpperCase();
    const originalStatus = (current?.original_status || current?.["match.status"] || current?.status || "").toUpperCase();
    try{
      await api.del("/api/manual-status",{params:{row_id: rowKey}});
    }catch(e){
      console.error("Falha ao remover override manual", e);
      showErrorToast(manualRemoveErrorTitle, e);
      return;
    }
    let updatedRow = null;
    let nextItems = null;
    setItems(prev=>{
      const mapped = prev.map(item=>{
        const key = resolveRowId(item);
        if(key !== rowKey) return item;
        const motivos = (item.motivos || "").split(";").filter(Boolean).filter(tag=>tag !== "ajuste_manual");
        const baseStatus = (item.original_status || item["match.status"] || item.status || originalStatus || "").toUpperCase();
        const updated = {
          ...item,
          status: baseStatus,
          "match.status": baseStatus,
          motivos: motivos.join(";"),
          _manual: false
        };
        delete updated.original_status;
        updatedRow = updated;
        return updated;
      });
      nextItems = mapped;
      return mapped;
    });
    if(updatedRow){ setSelectedRow(updatedRow); }
    if(nextItems){
      persistGridState({
        items: nextItems,
        total,
        filters,
        sortBy,
        sortDir,
        limit,
        offset
      });
    }
    if(previousStatus && originalStatus){ updateStats(previousStatus, originalStatus); }
  };

  useEffect(()=>{
    if(!currentJobId){
      return;
    }
    let cancelled = false;
    let timerId = null;
    const poll = async ()=>{
      try{
        const res = await api.get(`/api/process/${currentJobId}`);
        if(cancelled){
          return;
        }
        setJobStatus(res);
        const status = res?.status || null;
        const normalized = String(status||"").toLowerCase();
        if(normalized === "success"){
          await Promise.all([loadMeta(), loadGrid(0)]);
          if(cancelled){
            return;
          }
          if(timerId){ clearInterval(timerId); timerId = null; }
          setCurrentJobId(null);
          setJobStatus(null);
          setJobCanceling(false);
        }else if(normalized === "failed" || normalized === "error"){
          if(timerId){ clearInterval(timerId); timerId = null; }
          setCurrentJobId(null);
          setJobStatus(null);
          setJobCanceling(false);
          if(!cancelled){
            const lastLog = Array.isArray(res?.logs) && res.logs.length ? res.logs[res.logs.length-1] : null;
            const message = res?.message || lastLog?.message || lastLog?.label;
            showErrorToast(processingFailedTitle, message || processingFailedMessage);
          }
        }else if(normalized === "cancelled"){
          if(timerId){ clearInterval(timerId); timerId = null; }
          setCurrentJobId(null);
          setJobStatus(null);
          setJobCanceling(false);
          if(!cancelled){
            showWarningToast(processingCancelledTitle, processingCancelledMessage);
          }
        }
      }catch(pollErr){
        console.error("Falha no polling do processamento", pollErr);
      }
    };
    poll();
    timerId = setInterval(poll, 4000);
    return ()=>{
      cancelled = true;
      if(timerId){ clearInterval(timerId); }
    };
  },[api, currentJobId, loadGrid, loadMeta, processingCancelledMessage, processingCancelledTitle, processingFailedMessage, processingFailedTitle, showErrorToast, showWarningToast]);

  const toolbarStats = useMemo(()=> computeToolbarStats(meta),[meta]);
  const hasActiveFilters = useMemo(()=>{
    const current = filters || {};
    return Object.values(current).some(value=> Boolean(value));
  },[filters]);
  const emptyStateMessage = hasActiveFilters ? emptyFilteredMessage : emptyDefaultMessage;
  const emptyStateHint = hasActiveFilters ? emptyFilteredHint : emptyDefaultHint;
  const subtitleChildren = useMemo(()=>{
    const parts = subtitleTemplate.split("{file}");
    const nodes = [];
    parts.forEach((part,index)=>{
      if(part){
        nodes.push(part);
      }
      if(index < parts.length - 1){
        nodes.push(React.createElement("code",{key:`subtitle-file-${index}`}, subtitleFile));
      }
    });
    return nodes;
  },[subtitleFile, subtitleTemplate]);

  return (

    React.createElement("div",{className:"max-w-[1400px] mx-auto p-4 md:p-6"},
      React.createElement("div",{className:"mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between"},
        React.createElement("div",{className:"space-y-1"},
          React.createElement("h2",{className:"text-2xl font-semibold"},appTitle),
          React.createElement("p",{className:"text-slate-600"},...subtitleChildren)
        ),
        React.createElement("div",{className:"flex gap-2"},
          React.createElement("button",{
            onClick:openUploadModal,
            disabled:uploading || jobRunning,
            className:"rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-soft hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
          }, (uploading || jobRunning)?processButtonProcessing:processButtonLabel)
        )
      ),
      currentJobId && jobStatus ? React.createElement(JobProgressPanel,{
        jobId: currentJobId,
        status: jobStatus,
        apiBase: DEFAULT_API_BASE,
        onCancel: handleCancelJob,
        canceling: jobCanceling
      }) : null,
      React.createElement(Toolbar,{
        stats: toolbarStats ?? computeToolbarStats(meta),
        filters,
        onStatusFilter: handleStatusFilter,
        onFilterChange: handleFilterChange,
        onReload:()=> loadGrid(0),
        onExportJson: exportJ,
        onExportX: exportX,
        onExportP: exportP,
        onClearFilters: handleClearFilters,
        onClearData: handleClearData,
        clearing: clearingData,
        loading,
        disabled: jobRunning
      }),
      error && React.createElement("div",{className:"p-3 rounded bg-rose-100 text-rose-800 mb-3",role:"alert"}, String(error)),
      React.createElement(Table,{
        columns: tableColumns,
        items,
        resolveRowId,
        selectedRowId,
        sortBy,
        setSortBy,
        sortDir,
        setSortDir,
        onInspect: handleInspect,
        onManualStatus: handleManualStatus,
        onResetStatus: handleResetStatus,
        onSortChange: handleSortChange,
        loading,
        emptyMessage: emptyStateMessage,
        emptyHint: emptyStateHint
      }),
      React.createElement(InspectorPanel,{
        row: selectedRow,
        onClose: ()=>setSelectedRow(null),
        onMark: handleManualStatus,
        onReset: handleResetStatus
      }),
      React.createElement(Paginator,{offset,setOffset,limit,setLimit,total,go:loadGrid}),
      React.createElement(UploadModal,{
        open: showUploadModal,
        onClose: closeUploadModal,
        onConfirm: handleUploadConfirm,
        submitting: uploading,
        error: uploadError
      })
    )
  );
}

function startApp(){
  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(React.createElement(ToastProvider,null,React.createElement(App)));
}

loadTranslations(I18N_DEFAULT_LOCALE).finally(()=>{
  startApp();
});
