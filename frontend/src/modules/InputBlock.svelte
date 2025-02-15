<script lang="ts">
	import ThemeInputField from '../components/ThemeInputField.svelte';
	import ThemeButton from '../components/ThemeButton.svelte';
	import { data } from '../state.svelte';
	import { outputCallback } from '../callbacks.svelte';

	let { inpEventCb = null, value = $bindable() } = $props();

	let typing: any = null;

	const randomInpText = () => {
		const ex1 = 'alpha beta gamma delta eta zeta epsilon'
		let textlen = 1
		if(typing) return
		typing = setInterval(() => {
			if(textlen > ex1.length) {
				clearInt();
			}
			console.log(textlen)
			inpEventCb(ex1.slice(0, textlen++))
		}, 100)
		const clearInt = () => {clearInterval(typing); typing = null}
	};
</script>

<div class="flex flex-row items-center justify-between space-x-2 xl:space-x-6">
	<ThemeInputField {inpEventCb} {value} maxlen={60} />
	<ThemeButton label="Generate" clickEventCb={outputCallback} />
	<ThemeButton label="Clear" clickEventCb={() => inpEventCb('')} />
	<ThemeButton label="Example" clickEventCb={randomInpText} />
</div>
