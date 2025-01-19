<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { activeComponent, data } from '../state.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	let inpText: string = $state('');
	// let tokenProbMappings: { name: string, prob: number }[] = [];

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};

	const runModel = async () => {
		try {
			return await fetch('/model/run', {
				method: 'POST',
				body: JSON.stringify({ text: inpText }),
				headers: {
					'content-Type': 'application/json'
				}
			})
				.then((res) => {
					activeComponent.name = 'Generate';
					res;
				})
				.catch((error) => console.log('Could not run the model ' + error));
		} catch (error) {
			console.log('Unable to fetch ' + error);
			return;
		}
	}

	const getDist = async () => {
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
	}

	const runAndGetDist = async () => {
		await runModel();
		await getDist();
	}

	$effect(() => {
		$inspect(data.tokenProbMappings);
	})
</script>

<div>
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={runAndGetDist} />
	<ThemeButton label="Clear" clickEventCb={() => (value = '')} />
	<ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
