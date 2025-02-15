import axios from 'axios';
import { data, active_model, global_state, input, activeComponent, params } from '../state.svelte';

// This function will load the model of name passed as param, fallback is to the default model on active_model on state.svelte
export const loadModel = async (model_name: string = active_model.model_name) => {
	return await axios
		.post('/model/load', { model_name })
		.then((res) => {
			global_state.isModelLoaded = true;
			return res.data;
		})
		.catch(error => {global_state.info = error; console.log(error) });
};

export const runModel = async (input_text: string) => {
	return await axios
		.post('/model/run', { text: input_text })
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });
};

export const getAct = async (act_name: string, layer_name: string | null, block: number | null) => {
	const res = await axios
		.post('/ckpt/act', { act_name, layer_name, block })
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });

	const data = res;
	console.log(data);

	if (act_name === 'embed' || act_name === 'pos_embed') {
		let embedOutput: ScatterPlotData = [];

		for (let i = 0; i < data.length; i++) {
			embedOutput.push({ x: data[i][0], y: data[i][1], token: global_state.tokens[i] });
		}

		global_state.embed_output = embedOutput;
	} else if (act_name === 'pattern') {
		let attnPatterns: HeatMapData[] = [];

		for (let i = 0; i < data.length; i++) {
			let attnPattern: HeatMapData = [];
			for (let j = 0; j < data[i].length; j++) {
				for (let k = 0; k < data[i][j].length; k++) {
					attnPattern.push({
						x: k,
						y: j,
						source: global_state.tokens[k],
						destination: global_state.tokens[j],
						score: data[i][j][k]
					});
				}
			}
			attnPatterns.push(attnPattern);
		}

		global_state.attn_patterns = attnPatterns;
	} else if (act_name === 'resid_pre') {
		global_state.ln_pre = data;
	} else {
		global_state.data = data;
	}
};

export const getProbDensity = async (
	act_name: string | null,
	layer_name: string | null,
	block: number
) => {
	const res = await axios
		.post('/ckpt/prob_density', { act_name, layer_name, block })
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });

	const data = res;
	console.log(data);

	if (act_name == 'resid_pre' || act_name === 'resid_mid') {
		global_state.ln_pre = data;
	} else if (act_name == 'normalized') {
		global_state.ln_post = data;
	}
};

// export const getAttnPattern = async () => {
// 	if (input.isChanged === true) {
// 		await runModel(input.text);
// 	}

// 	await getAct("pattern", "attn", global_state.active_block);
// 	activeComponent.name = "attn";
// }

export const getMLPOuts = async (
	act_name: string,
	layer_name: string | null,
	block: number | null,
	neuron: number | null
) => {
	const res = await axios
		.post('/ckpt/mlp_outs', { act_name, layer_name, block, neuron })
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });

	const data = res;
	console.log(data);
	let act = [];

	for (let i = 0; i < data.length; i++) {
		act.push({
			token: global_state.tokens[i],
			score: data[i]
		});
	}

	console.log(act);

	global_state.data = act;
};

export const getDist = async () => {
	const res = await axios
		.post('/model/dist', {temperature: params.temperature})
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });

	data.tokenProbMappings = res;
};

export const getNextToken = async () => {
	const res = await axios
		.post("/model/sample", { temperature: params.temperature, p: params.top_p, k: params.top_k })
		.then((res) => res.data["next_token"])
		.catch(error => {global_state.info = error; console.log(error) });

	console.log(res);
	global_state.next_token_id = res;
}

export const getTokens = async (input_text: string) => {
	const res = await axios
		.post('/model/tokenize', { text: input_text })
		.then((res) => res.data)
		.catch(error => {global_state.info = error; console.log(error) });

	global_state.tokens = res;
};
