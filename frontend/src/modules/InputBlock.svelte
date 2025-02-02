<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { data } from '../state.svelte';
	import { outputCallback } from '../callbacks.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	const randomInpText = () => {
		inpEventCb('alpha beta gamma delta eta zeta epsilon');
	};

	$effect(() => {
		$inspect(data.tokenProbMappings);
	});
</script>

<div class="flex flex-row items-center justify-between space-x-2 xl:space-x-6">
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={outputCallback} />
	<ThemeButton label="Clear" clickEventCb={() => inpEventCb('')} />
	<ThemeButton label="Example" clickEventCb={randomInpText} />
</div>
