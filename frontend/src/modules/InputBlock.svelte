<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { activeComponent } from '../state.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	let inpText: string = $state('');

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
					res
				})
				.catch((error) => console.log("Could not run the model " + error));
		} catch (error) {
			console.log("Unable to fetch " + error);
			return;
		}
	}
</script>

<div>
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={runModel} />
	<ThemeButton label="Clear" clickEventCb={() => (value = '')} />
	<ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
