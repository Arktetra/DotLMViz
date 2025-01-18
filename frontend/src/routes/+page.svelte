<script lang="ts">
    import OutputBlock from "../modules/OutputBlock.svelte";
    import TransformerBlocks from "../modules/TransformerBlocks.svelte";
    import UnembeddingBlock from "../modules/UnembeddingBlock.svelte";
    import EmbeddingBlock from "../modules/EmbeddingBlock.svelte";
    import InputBlock from "../modules/InputBlock.svelte";
    import TokensBlock from "../modules/TokensBlock.svelte";

    import Navbar from "../lib/Navbar.svelte";
    import DottedBlockBase from "../components/DottedBlockBase.svelte";
	import ThemeButton from "../components/ThemeButton.svelte";

    let tokens: string[] = $state([])
    let inpText: string = $state("")
    let activeTokenInd: number = $state(0)

    const onInpChange = (v : string) => {
        inpText = v
        genToken()
    }

    $effect(() => {
        genToken()
    })

    const genToken = () => {
        // tokens = inpText.indexOf(' ') > 0 || inpText.length > 5 ? inpText.split(' ') : inpText.split('')
        tokens = inpText.split(' ')
    }
</script>
  
<Navbar />
<section class="min-w-[1500px] max-h-screen min-h-[900px] flex flex-col justify-evenly items-center">
    <div class="flex flex-row justify-evenly items-center space-x-10">
        <TokensBlock {tokens} bind:tokenInd={activeTokenInd}>
            <span class="text-sm font-light text-theme-w">Index: <span class="text-md font-bold">{activeTokenInd}</span></span>
            <span class="text-sm font-light text-theme-w">Token: <span class="text-lg font-bold">'{tokens[activeTokenInd]}'</span></span>
        </TokensBlock>
        <DottedBlockBase label="GPT-2 Small" titStyle="text-xl font-bold" borderSize={"1px"} inStyle="min-w-[50%] m-2 py-2 pt-8 flex-row justify-between space-x-10">
            <EmbeddingBlock />
            <TransformerBlocks />
            <UnembeddingBlock />
        </DottedBlockBase>
        <DottedBlockBase label="Output">
            <div class="min-w-[5rem] min-h-[5rem] flex flex-col justify-evenly items-center">
                <span class="bg-theme rounded-md p-1 px-2 text-theme-w font-light">E</span>
            </div>
        </DottedBlockBase>
    </div>

    
    <InputBlock bind:value={inpText} inpEventCb={onInpChange} />
</section>
<OutputBlock />