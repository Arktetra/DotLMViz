<script lang="ts">
	import { scale } from 'svelte/transition';

	const { label = '', children = null, clickEventCb = null } = $props();

	const btnClick = () => {
		if (clickEventCb) clickEventCb();
	};
</script>

<button
	onclick={btnClick}
	transition:scale={{ duration: 400 }}
	class="min-h-8 min-w-8 m-2 p-1 px-2 rounded-md bg-theme text-theme-w transition-colors duration-300 hover:bg-theme-alt"
>
	{label}
	{#if children}
		{@render children()}
	{/if}
</button>
