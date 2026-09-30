use neat_ai_discovery::analysis::detection::topology_diversification::detect_topology_diversification_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::time::Instant;
fn n(u:&str,t:&str)->NeuronJson{NeuronJson{uuid:u.into(),neuron_type:t.into(),squash:"IDENTITY".into(),bias:0.0}}
fn s(a:&str,b:&str)->SynapseJson{SynapseJson{from_uuid:a.into(),to_uuid:b.into(),weight:0.5,synapse_type:None}}
fn recs()->Vec<(String,Vec<DiscoverRecord>)>{vec![("out".into(),(0..20).map(|i|DiscoverRecord{obs_index:i,neuron_uuid:"out".into(),value:Some(0.0),activation:0.0,errors:vec![1.0]}).collect())]}
fn ladder(l:usize)->CreatureJson{
 let mut ns=vec![n("in","input")];let mut ss=vec![];
 let mut prev=vec!["in".to_string()];
 for i in 0..l{let cur=vec![format!("a{i}"),format!("b{i}")];for c in &cur{ns.push(n(c,"hidden"));for p in &prev{ss.push(s(p,c));}}prev=cur;}
 ns.push(n("out","output"));for p in &prev{ss.push(s(p,"out"));}
 CreatureJson{neurons:ns,synapses:ss,input:1,output:1, ..Default::default()}}
fn chain(l:usize)->CreatureJson{
 let mut ns=vec![n("in","input")];let mut ss=vec![];let mut prev="in".to_string();
 for i in 0..l{let c=format!("h{i}");ns.push(n(&c,"hidden"));ss.push(s(&prev,&c));prev=c;}
 ns.push(n("out","output"));ss.push(s(&prev,"out"));
 CreatureJson{neurons:ns,synapses:ss,input:1,output:1, ..Default::default()}}
#[test] fn probe_time(){let r=recs();for l in [4,8,12,16]{let c=ladder(l);let t=Instant::now();let o=detect_topology_diversification_candidates(&c,&r);println!("ladder {l}: {:?} cands={}",t.elapsed(),o.len());}}
#[test] fn probe_stack(){let r=recs();let c=chain(50_000);let h=std::thread::Builder::new().stack_size(2*1024*1024).spawn(move||detect_topology_diversification_candidates(&c,&r).len()).unwrap();println!("chain ok {:?}",h.join());}
