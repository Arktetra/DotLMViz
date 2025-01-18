<script lang="ts">
	import { QuestionCircleSolid } from 'flowbite-svelte-icons';

	import SideDrawer from '../components/SideDrawer.svelte';
	import ThemeInputSlider from '../components/ThemeInputSlider.svelte';
</script>

<SideDrawer width={'25rem'}>
	<div class="flex h-full w-full flex-col items-center justify-evenly">
		<div class="w-full rounded-md bg-theme-g p-2 shadow-inner shadow-theme-g-alt relative">
			<a href="/read/control-parameter" title="Control Parameter" class="absolute end-1 top-1 text-theme">
				<QuestionCircleSolid size={'sm'} />
			</a>
			<span class="text-md mb-2 block text-center font-extrabold uppercase text-theme underline">Control Parameters</span>
			<ThemeInputSlider label={'Temperature'} min={-2} max={2} step={0.1} />
			<hr class="my-1 border border-theme-w" />
			<ThemeInputSlider label={'Top K'} min={1} max={10} step={1} />
			<hr class="my-1 border border-theme-w" />
			<ThemeInputSlider label={'Top P'} min={0} max={1} step={0.05} />
		</div>
		<hr class="w-full border border-theme" />
		<h1 class="text-md my-2 text-center font-extrabold uppercase text-theme">Softmax Output</h1>
		<div
			class="flex min-h-[15rem] w-full flex-col items-center justify-evenly rounded-md bg-theme-g-alt p-3 shadow-inner shadow-theme-g-alt"
		>
			<div class="chart w-full text-right text-ti font-light">
				<span class="text-md my-4 block text-center font-bold text-theme underline"
					>Chart here.. Probability distribution</span
				>
			</div>
		</div>
		<span class="my-2 font-bold text-theme"
			>Next Token : <span class="rounded-md bg-theme p-1 px-2 font-light text-theme-w">E</span
			></span
		>
	</div>
</SideDrawer>

<style lang="css">
	.chart :global(div) {
		background-color: #665191;
		padding: 3px;
		margin: 1px;
		color: white;
		opacity: 50%;
	}

	.chart :global(div):hover {
		opacity: 100%;
	}
</style>
