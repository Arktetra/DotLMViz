import { data, active_model, global_state, input, activeComponent } from '../state.svelte';

// This function will load the model of name passed as param, fallback is to the default model on active_model on state.svelte
export const loadModel = async (model_name: string = active_model.model_name) => {
	try {
		return await fetch('/model/load', {
			method: 'POST',
			body: JSON.stringify({ model_name }),
			headers: {
				'Content-Type': 'application/json'
			}
		})
			.then((res) => {
				global_state.isModelLoaded = true;
				res
			})
			.catch((error) => console.log('Something not right ' + error));
	} catch (error) {
		console.log('Unable to fetch ' + error);
		return;
	}
};

export const runModel = async (input_text: string) => {
	try {
		const response = await fetch('/model/run', {
			method: 'POST',
			body: JSON.stringify({ text: input_text }),
			headers: {
				'Content-Type': 'application/json'
			}
		});

		if (!response.ok) {
			throw new Error(`${response.status}, ${response.statusText}`);
		}

		return response;
	} catch (error) {
		throw error;
	}
};

export const getAct = async (act_name: string, layer_name: string | null, block: number | null) => {
	try {
		const res = await fetch('/ckpt/act', {
			method: 'POST',
			body: JSON.stringify({ act_name, layer_name, block }),
			headers: {
				'Content-Type': 'application/json'
			}
		})

		if (!res.ok) {
			throw new Error(`Response status: ${res.status}`);
		}

		let data = await res.json();

		console.log(data);

		if (act_name === "embed" || act_name === "pos_embed") {
			let embedOutput: ScatterPlotData = [];

			for (let i = 0; i < data.length; i++) {
				embedOutput.push({x: data[i][0], y: data[i][1], token: global_state.tokens[i]});
			}

			global_state.embed_output = embedOutput;
		} else if (act_name === "pattern") {
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
						})
					}
				}
				attnPatterns.push(attnPattern);
			}

			global_state.attn_patterns = attnPatterns;
		} else if (act_name === "resid_pre") {
			global_state.ln_pre = data;
		} else {
			global_state.data = data;
		}

	} catch (error: any) {
		console.log(error.message);
		return;
	}
};

export const getLN1PreAct = async (act_name: string | null, layer_name: string | null, block: number) => {
	if (input.isChanged === true) {
		await runModel(input.text);
	}

	try {
		const res = await fetch('/ckpt/prob_density', {
			method: 'POST',
			body: JSON.stringify({ act_name, layer_name, block }),
			headers: {
				'Content-Type': 'application/json'
			}
		})

		if (!res.ok) {
			throw new Error(`Response status: ${res.status}`);
		}

		let data = await res.json();

		console.log(data);
	} catch (error: any) {
		console.log(error.message);
		return;
	}
}

export const getAttnPattern = async () => {
	if (input.isChanged === true) {
		await runModel(input.text);
	}

	await getAct("pattern", "attn", global_state.active_block);
	activeComponent.name = "attn";
}

export const getMLPOuts = async (act_name: string, layer_name: string | null, block: number | null, neuron: number | null) => {
	try {
		const res = await fetch('/ckpt/mlp_outs', {
			method: 'POST',
			body: JSON.stringify({ act_name, layer_name, block, neuron }),
			headers: {
				'Content-Type': 'application/json'
			}
		})

		if (!res.ok) {
			throw new Error(`Response status: ${res.status}`);
		}

		let data = await res.json();

		let act = []

		for (let i = 0; i < data.length; i++) {
			act.push({
				token: global_state.tokens[i],
				score: data[i]
			})
		}

		console.log(act);

		global_state.data = act;
	} catch (error: any) {
		console.log(error.message);
		return;
	}
};

export const getMLPPre = async () => {
	if (input.isChanged === true) {
		await runModel(input.text);
	}

	await getMLPOuts("pre", "mlp", 0, global_state.neuron);
	activeComponent.name = "mlp_pre";
}

export const getDist = async () => {
	try {
		const res = await fetch('/model/dist');

		if (!res.ok) {
			throw new Error(`Response status: ${res.status}`);
		}

		data.tokenProbMappings = await res.json();
	} catch (error: any) {
		console.error(error.message);
		return;
	}
};

export const getTokens = async (input_text: string) => {
	try {
		const response = await fetch('/model/tokenize', {
			method: "POST",
			body: JSON.stringify({ text: input_text }),
			headers: {
				'Content-Type': 'application/json'
			}
		})

		if (!response.ok) {
			if (response.status != 500) {
				throw new Error(`${response.status}, ${response.statusText}`);
			}
		}

		let tokens = await response.json();

		global_state.tokens = tokens;
	} catch (error: any) {
		console.error(error.message);

		return;
	}
}